"""Summarise the pilot (default config per method at n=2,000 and n=8,000 on one dataset,
validation only, cost mode) and estimate the wall time of every remaining stage.

* ``results_pilot/pilot_summary.csv``   per (method, size): fit time, epochs, memory, latency
* ``configs/scheduler_estimates.json``  per (method, size): fit_s, peak_rss_mb, peak_vram_mb
                                        (scheduler ordering + memory guard; sizes in between
                                        are interpolated linearly)
* ``results_pilot/pilot_estimate.csv``  estimated hours per stage

Run time of a job = its measured wall time (fit + latency + start-up), interpolated in
size between the two pilot sizes. Exclusive stages (tuning, cost pass) are the plain sum;
parallel quality stages are divided over the lanes (``k_gpu`` per GPU for neural jobs,
``cpu_jobs`` for classical jobs) with ``--slowdown`` per-job slowdown under concurrency, and
the stage takes as long as its slower lane. Quality runs skip latency, so their time is
the fit time plus start-up.
Usage: ``python -m ksent.analysis.pilot_report [--k-gpu K --cpu-jobs C --slowdown 1.2]``
"""
from __future__ import annotations

import argparse
import glob
import json

import pandas as pd

from ksent.config import SOURCE_DIR, family, global_cfg
from ksent.utils import read_json, write_json

CHAR_SLOWDOWN = 1.5  # char sequences are longer than Kiwi morpheme sequences
STARTUP_S = 6.0      # interpreter + imports + data loading per subprocess (measured: wall - fit - latency)


def load(root) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(str(root / "raw" / "*" / "*" / "*" / "*.json"))):
        if p.endswith(".failed.json"):
            continue
        r = read_json(p)
        lat = r.get("latency") or {}
        rows.append({
            "dataset": r["dataset"], "method": r["method"], "size": r["size"], "device": r["train_device"],
            "mamba_backend": r.get("mamba_backend"), "val_f1": round(r["val"]["macro_f1"] * 100, 2),
            "fit_e2e_s": r["cost"]["fit_time_end_to_end_s"], "epochs": (r.get("fit") or {}).get("epochs_run"),
            "lat_cpu_e2e_ms": lat.get("cpu", {}).get("end_to_end", {}).get("median_ms"),
            "lat_acc_e2e_ms": lat.get("accelerator", {}).get("end_to_end", {}).get("median_ms"),
            "peak_rss_mb": r["cost"]["peak_rss_mb"], "peak_acc_mb": r["cost"].get("peak_accelerator_mem_train_mb"),
            "peak_acc_infer_mb": r["cost"].get("peak_accelerator_mem_infer_mb"),
            "serialized_mb": r["cost"].get("serialized_mb"), "wall_s": r["wall_time_s"],
            "timing_reportable": r.get("timing_reportable"),
        })
    return pd.DataFrame(rows)


def _interp(df: pd.DataFrame, method: str, size: int, col: str) -> float:
    pts = df[df.method == method].sort_values("size")[["size", col]].dropna().to_numpy()
    if len(pts) == 0:
        return float("nan")
    if len(pts) == 1:
        return float(pts[0][1] * size / pts[0][0])
    (n0, v0), (n1, v1) = pts[0], pts[-1]
    return float(max(0.0, v0 + (v1 - v0) * (size - n0) / (n1 - n0)))


def main() -> None:
    g = global_cfg()
    sc = g.get("scheduler") or {}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k-gpu", type=int, default=sc.get("k_gpu") or 1)
    ap.add_argument("--cpu-jobs", type=int, default=sc.get("cpu_jobs") or 6)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--slowdown", type=float, default=1.2, help="per-job slowdown under concurrency")
    args = ap.parse_args()

    df = load(SOURCE_DIR / "results_pilot")
    if df.empty:
        print("no pilot results found in results_pilot/")
        return
    pd.set_option("display.width", 240)
    print(df.drop(columns=["dataset"]).round(3).to_string(index=False))
    alt = load(SOURCE_DIR / "results_pilot" / "mamba_torch_ref")
    if len(alt):
        for n in sorted(alt["size"].unique()):
            a = df[(df.method == "small_mamba") & (df["size"] == n)]
            b = alt[alt["size"] == n]
            if len(a) and len(b):
                a, b = a.iloc[0], b.iloc[0]
                print(f"\nMamba backend check n={n}: {a.mamba_backend}: fit {a.fit_e2e_s:.1f}s, acc lat "
                      f"{a.lat_acc_e2e_ms} ms, val F1 {a.val_f1} | torch_ref: fit {b.fit_e2e_s:.1f}s, acc lat "
                      f"{b.lat_acc_e2e_ms} ms, val F1 {b.val_f1}")

    # ---- scheduler estimates (method -> size -> values)
    est: dict = {}
    for _, r in df.iterrows():
        est.setdefault(r.method, {})[str(int(r["size"]))] = {
            "fit_s": float(r.fit_e2e_s), "wall_s": float(r.wall_s), "peak_rss_mb": float(r.peak_rss_mb),
            "peak_vram_mb": float(max(r.peak_acc_mb or 0, r.peak_acc_infer_mb or 0)) if family(r.method) == "neural"
            else None}
    write_json(est, SOURCE_DIR / "configs" / "scheduler_estimates.json")

    # ---- per-job time models
    lat_s = (df.wall_s - df.fit_e2e_s).clip(lower=0)  # latency + start-up in cost runs
    df = df.assign(cost_wall_s=df.wall_s, quality_wall_s=df.fit_e2e_s + STARTUP_S, lat_overhead_s=lat_s)

    def t_job(m: str, n: int, tok: str = "kiwi", quality: bool = False) -> float:
        col = "quality_wall_s" if quality else "cost_wall_s"
        v = _interp(df, m, n, col)
        return v * (CHAR_SLOWDOWN if tok == "char" and family(m) == "neural" else 1.0)

    D, S, M = len(g["datasets"]), [int(s) for s in g["sizes"]], list(g["methods"])
    seeds, n_tune = len(g["seeds"]), int(g["tuning_size"])
    tfidf_m = [m for m in M if family(m) == "tfidf"]
    neural_m = [m for m in M if family(m) == "neural"]

    def parallel_h(jobs: list[tuple[str, int, str]]) -> float:
        cpu = sum(t_job(m, n, t, True) for m, n, t in jobs if family(m) == "tfidf") * args.slowdown / args.cpu_jobs
        gpu = sum(t_job(m, n, t, True) for m, n, t in jobs if family(m) == "neural") * args.slowdown \
            / (args.k_gpu * args.gpus)
        return max(cpu, gpu) / 3600

    # tuning runs measure no latency: fit + start-up, exclusive and sequential
    tuning_h = sum(_interp(df, m, n_tune, "fit_e2e_s") + STARTUP_S for m in M) * 10 * D / 3600
    tuning_q_h = parallel_h([(m, n_tune, "kiwi") for m in M for _ in range(10 * D)])
    sweep_h = parallel_h([(m, n, "kiwi") for m in M for n in S for _ in range(seeds * D)])
    # ablation: worst-case pair (slowest TF-IDF, slowest neural) x 2 new tokenizers x {500, 8000}
    ab = g["ablation"]
    wt = max(tfidf_m, key=lambda m: t_job(m, 8000, quality=True))
    wn = max(neural_m, key=lambda m: t_job(m, 8000, quality=True))
    ab_jobs = [(m, int(n), t) for m in (wt, wn) for t in ab["tokenizers"] if t != g["tokenizer"]
               for n in ab["sizes"] for _ in range(seeds * D)]
    ablation_h = parallel_h(ab_jobs)
    cost8k_h = sum(t_job(m, 8000) for m in M) * seeds * D / 3600
    cross_h = (t_job(wt, 4000) + t_job(wn, 4000)) * seeds * D / 3600     # upper bound: all 3 datasets at 4,000
    total = tuning_h + sweep_h + ablation_h + cost8k_h + cross_h
    est_rows = [("D tuning (exclusive, cost mode)", tuning_h), ("D tuning (quality route, parallel)", tuning_q_h),
                ("E main sweep (quality, parallel)", sweep_h), ("F ablation (quality, parallel, upper bound)", ablation_h),
                ("G1 cost pass n=8,000 (exclusive)", cost8k_h), ("G2 crossover cost (exclusive, upper bound)", cross_h),
                ("TOTAL (tuning exclusive)", total)]
    out = pd.DataFrame(est_rows, columns=["stage", "hours"])
    print(f"\nEstimated wall time (k_gpu={args.k_gpu}, cpu_jobs={args.cpu_jobs}, slowdown={args.slowdown}):")
    print(out.round(2).to_string(index=False))
    print("\nPer-job cost-mode wall time (s), interpolated:")
    print(pd.DataFrame({n: {m: round(t_job(m, n), 1) for m in M} for n in S}).to_string())
    df.to_csv(SOURCE_DIR / "results_pilot" / "pilot_summary.csv", index=False)
    out.to_csv(SOURCE_DIR / "results_pilot" / "pilot_estimate.csv", index=False)
    print(json.dumps({"tuning_route": "quality+replay" if tuning_h > 8 else "exclusive"}))


if __name__ == "__main__":
    main()
