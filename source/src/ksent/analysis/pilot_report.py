"""Summarise the pilot (default config, n=2,000, one dataset) and estimate runtimes for
tuning, the main sweep and the tokenizer ablation.

Estimates assume fit time scales linearly with training size (an upper bound for
early-stopped neural models, roughly right for the rest) and add each run's measured
non-fit overhead (latency measurement, I/O, start-up). The ablation estimate uses the
slowest TF-IDF and the slowest neural method (the actual pair is chosen later) and a
factor for longer character sequences.
Usage: ``python -m ksent.analysis.pilot_report``
"""
from __future__ import annotations

import glob

import pandas as pd

from ksent.config import SOURCE_DIR, family, global_cfg
from ksent.utils import read_json

CHAR_SLOWDOWN = 1.5  # rough: char sequences are longer than Kiwi morpheme sequences


def load(root) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(str(root / "raw" / "*" / "*" / "*" / "*.json"))):
        if p.endswith(".failed.json"):
            continue
        r = read_json(p)
        lat = r.get("latency") or {}
        rows.append({
            "dataset": r["dataset"], "method": r["method"], "device": r["train_device"],
            "mamba_backend": r.get("mamba_backend"), "val_f1": round(r["val"]["macro_f1"] * 100, 2),
            "fit_e2e_s": r["cost"]["fit_time_end_to_end_s"], "epochs": (r.get("fit") or {}).get("epochs_run"),
            "lat_cpu_e2e_ms": lat.get("cpu", {}).get("end_to_end", {}).get("median_ms"),
            "lat_acc_e2e_ms": lat.get("accelerator", {}).get("end_to_end", {}).get("median_ms"),
            "peak_rss_mb": r["cost"]["peak_rss_mb"], "peak_acc_mb": r["cost"].get("peak_accelerator_mem_train_mb"),
            "serialized_mb": r["cost"].get("serialized_mb"), "wall_s": r["wall_time_s"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    g = global_cfg()
    df = load(SOURCE_DIR / "results_pilot")
    if df.empty:
        print("no pilot results found in results_pilot/")
        return
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))
    alt = load(SOURCE_DIR / "results_pilot" / "mamba_torch_ref")
    if len(alt):
        a = df[df.method == "small_mamba"].iloc[0]
        b = alt.iloc[0]
        print(f"\nMamba backend check: {a.mamba_backend}: fit {a.fit_e2e_s:.1f}s, acc latency {a.lat_acc_e2e_ms} ms, "
              f"val F1 {a.val_f1} | torch_ref: fit {b.fit_e2e_s:.1f}s, acc latency {b.lat_acc_e2e_ms} ms, val F1 {b.val_f1}")
        print("-> choose ONE backend and set `mamba_backend:` in configs/global.yaml before tuning.")
    n_tune = int(g["tuning_size"])
    overhead = (df.wall_s - df.fit_e2e_s).clip(lower=0)
    tune = (df.fit_e2e_s + 3.0) * 10 * len(g["datasets"])
    scale = sum(n / n_tune for n in g["sizes"])
    sweep = (df.fit_e2e_s * scale + overhead * len(g["sizes"])) * len(g["seeds"]) * len(g["datasets"])
    est = pd.DataFrame({"method": df.method, "tuning_h": tune / 3600, "sweep_h": sweep / 3600})
    ab = g["ablation"]
    n_new_tok = len([t for t in ab["tokenizers"] if t != g["tokenizer"]])
    ab_scale = sum(n / n_tune for n in ab["sizes"])
    per = df.assign(run=df.fit_e2e_s * ab_scale + overhead * len(ab["sizes"]), fam=df.method.map(family))
    worst = per.groupby("fam").run.max()
    ablation_h = (worst.get("tfidf", 0) + worst.get("neural", 0) * CHAR_SLOWDOWN) * n_new_tok * len(g["seeds"]) \
        * len(g["datasets"]) / 3600
    print("\nEstimated runtime (hours, sequential):")
    print(est.round(2).to_string(index=False))
    print(f"TOTAL tuning ≈ {est.tuning_h.sum():.1f} h   main sweep ≈ {est.sweep_h.sum():.1f} h   "
          f"ablation ≤ {ablation_h:.1f} h (upper bound)")
    df.to_csv(SOURCE_DIR / "results_pilot" / "pilot_summary.csv", index=False)
    est.to_csv(SOURCE_DIR / "results_pilot" / "pilot_estimate.csv", index=False)


if __name__ == "__main__":
    main()
