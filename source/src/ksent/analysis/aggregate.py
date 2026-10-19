"""Aggregate per-run JSONs into tidy CSVs under ``results/aggregated/``.

* ``runs.csv``             one row per (dataset, method, tokenizer, size, seed)
* ``summary.csv``          main tokenizer only: mean and sample std (ddof=1) over seeds
* ``summary_tokenizers.csv`` all tokenizers (used by the ablation)
* ``confusion_n8000.csv``  main tokenizer, n=8,000: confusion counts summed over seeds + rates
* ``tuning.csv`` / ``frozen.csv``  tuning runs and selected configs
* ``meta.json``            hardware fingerprints, devices, Mamba backends, mixed-hardware flag

Runs from different hardware fingerprints are refused unless ``--allow-mixed-hardware``
(then a warning is printed and ``meta.json`` marks the tables as mixed).

Quality/cost split: predictive metrics (val/test, confusion, OOV, epochs) come only from
the quality runs in ``results/raw/``; every cost metric (fit time, latency, memory,
serialized size) comes only from the exclusive cost runs in ``results/cost/raw/``, joined
on (dataset, method, tokenizer, size, seed). Cells without a cost run have no cost values.
``runs_cost.csv`` holds the cost runs on their own.
"""
from __future__ import annotations

import argparse
import glob

import pandas as pd

from ksent.analysis.common import AGG_DIR
from ksent.config import global_cfg, load_yaml, main_tokenizer, path
from ksent.utils import read_json, write_json

ID_COLS = ["dataset", "method", "family", "tokenizer", "size", "seed"]


def flatten(r: dict) -> dict:
    """Flatten one result record into a tidy row."""
    row = {k: r.get(k) for k in ID_COLS}
    hw = r.get("hardware") or {}
    row.update(config_id=r.get("config_id"), train_device=r.get("train_device"), model_device=r.get("model_device"),
               mamba_backend=r.get("mamba_backend"),
               precision=r.get("precision"), n_threads=r.get("n_threads"), fingerprint=hw.get("fingerprint"),
               gpu_name=hw.get("gpu_name"), cpu_model=hw.get("cpu_model"),
               timing_reportable=r.get("timing_reportable", True), mps_fallback_used=r.get("mps_fallback_used"),
               git_commit=r.get("git_commit"))
    for split in ("val", "test"):
        m = r.get(split) or {}
        for k in ("macro_f1", "accuracy", "macro_precision", "macro_recall"):
            row[f"{split}_{k}"] = m.get(k)
        if m.get("confusion"):
            (tn, fp), (fn, tp) = m["confusion"]
            row.update({f"{split}_tn": tn, f"{split}_fp": fp, f"{split}_fn": fn, f"{split}_tp": tp})
    c = r.get("cost") or {}
    row.update(fit_e2e_s=c.get("fit_time_end_to_end_s"), fit_clf_s=c.get("fit_time_classifier_s"),
               tokenize_s=c.get("tokenize_time_s"), featurize_s=c.get("featurize_time_s"),
               peak_rss_mb=c.get("peak_rss_mb"), peak_rss_delta_mb=c.get("peak_rss_delta_mb"),
               serialized_mb=c.get("serialized_mb"), peak_acc_mb=c.get("peak_accelerator_mem_train_mb"))
    lat = r.get("latency") or {}
    for key, prefix in (("cpu", "lat_cpu"), ("accelerator", "lat_acc")):
        if key in lat:
            for kind, short in (("end_to_end", "e2e"), ("classifier_only", "clf")):
                row[f"{prefix}_{short}_ms"] = lat[key][kind]["median_ms"]
                row[f"{prefix}_{short}_p95_ms"] = lat[key][kind]["p95_ms"]
    row["acc_device"] = (lat.get("accelerator") or {}).get("classifier_only", {}).get("device")
    ms = r.get("model_size") or {}
    row.update(params_emb=ms.get("embedding"), params_nonemb=ms.get("non_embedding"),
               total_nodes=ms.get("total_nodes"), n_trees=ms.get("n_trees"), n_features=ms.get("n_features"),
               n_nonzero_coef=ms.get("n_nonzero_coef"), vocab_size=ms.get("vocab_size"))
    pp = r.get("preprocessing") or {}
    row.update(oov_token_rate=pp.get("oov_token_rate"), oov_review_rate=pp.get("oov_review_rate"),
               truncation_rate=pp.get("truncation_rate"))
    f = r.get("fit") or {}
    row.update(best_epoch=f.get("best_epoch"), epochs_run=f.get("epochs_run"), wall_s=r.get("wall_time_s"))
    return row


COST_COLS = ["fit_e2e_s", "fit_clf_s", "tokenize_s", "featurize_s", "peak_rss_mb", "peak_rss_delta_mb",
             "serialized_mb", "peak_acc_mb", "lat_cpu_e2e_ms", "lat_cpu_e2e_p95_ms", "lat_cpu_clf_ms",
             "lat_cpu_clf_p95_ms", "lat_acc_e2e_ms", "lat_acc_e2e_p95_ms", "lat_acc_clf_ms", "lat_acc_clf_p95_ms",
             "acc_device"]
JOIN = ["dataset", "method", "tokenizer", "size", "seed"]


def merge_cost(quality: pd.DataFrame, cost: pd.DataFrame) -> pd.DataFrame:
    """Quality rows with their cost columns replaced by the matching cost run's values."""
    q = quality.drop(columns=[c for c in COST_COLS if c in quality], errors="ignore")
    q["timing_reportable"] = False
    if cost.empty:
        return q
    c = cost[JOIN + [k for k in COST_COLS if k in cost] + ["timing_reportable"]].copy()
    c = c[c.timing_reportable.fillna(False).astype(bool)]
    out = q.drop(columns=["timing_reportable"]).merge(c, on=JOIN, how="left")
    out["timing_reportable"] = out["timing_reportable"].fillna(False).astype(bool)
    return out


def load_runs(root: str = "raw") -> pd.DataFrame:
    """All successful runs under ``results/<root>``."""
    rows = []
    for p in sorted(glob.glob(str(path("results") / root / "*" / "*" / "*" / "*.json"))):
        if p.endswith(".failed.json"):
            continue
        r = read_json(p)
        if r.get("status") == "ok":
            rows.append(flatten(r))
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame, keys: list[str] | None = None) -> pd.DataFrame:
    """Mean and sample std (ddof=1) over seeds for every numeric column."""
    keys = keys or ["dataset", "method", "family", "tokenizer", "size"]
    num = [c for c in runs.columns if c not in keys + ["seed", "config_id", "n_threads"]
           and pd.api.types.is_numeric_dtype(runs[c]) and runs[c].dtype != bool]
    g = runs.groupby(keys, sort=False)
    out = g[num].agg(["mean", lambda s: s.std(ddof=1)])
    out.columns = [f"{c}_{'mean' if a == 'mean' else 'std'}" for c, a in out.columns]
    out["n_seeds"] = g["seed"].nunique()
    out["all_timings_reportable"] = g["timing_reportable"].all()
    return out.reset_index()


def confusion_8000(runs: pd.DataFrame) -> pd.DataFrame:
    """Confusion counts summed over seeds at the largest size, then row-normalised."""
    n = max(int(s) for s in global_cfg()["sizes"])
    sub = runs[runs["size"] == n]
    if sub.empty or "test_tn" not in sub:
        return pd.DataFrame()
    cm = sub.groupby(["dataset", "method"])[["test_tn", "test_fp", "test_fn", "test_tp"]].sum().reset_index()
    cm["n_runs"] = sub.groupby(["dataset", "method"])["seed"].nunique().values
    neg, pos = cm.test_tn + cm.test_fp, cm.test_fn + cm.test_tp
    cm["tnr"], cm["fpr"], cm["fnr"], cm["tpr"] = cm.test_tn / neg, cm.test_fp / neg, cm.test_fn / pos, cm.test_tp / pos
    return cm


def load_tuning() -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(str(path("results") / "tuning" / "*" / "*" / "cfg*.json"))):
        if p.endswith(".failed.json"):
            continue
        r = read_json(p)
        if r.get("status") != "ok":
            continue
        rows.append({"dataset": r["dataset"], "method": r["method"], "config_id": r["config_id"],
                     "val_macro_f1": r["val"]["macro_f1"], "fit_e2e_s": r["cost"]["fit_time_end_to_end_s"],
                     "xgb_best_rounds": (r.get("fit") or {}).get("xgb_best_rounds"),
                     "fingerprint": (r.get("hardware") or {}).get("fingerprint"),
                     "timing_reportable": r.get("timing_reportable", True)})
    return pd.DataFrame(rows)


def load_frozen() -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(str(path("frozen") / "*" / "*.yaml"))):
        f = load_yaml(p)
        rows.append({"dataset": f["dataset"], "method": f["method"], "config_id": f["selected_config_id"],
                     "val_macro_f1": f["val_macro_f1"], "tuning_time_total_s": f["tuning_time_total_s"],
                     "params": repr(f["params"])})
    return pd.DataFrame(rows)


def hardware_check(frames: list[pd.DataFrame], allow_mixed: bool) -> dict:
    """Refuse (or flag) results coming from more than one hardware fingerprint."""
    fps = sorted({fp for df in frames if len(df) and "fingerprint" in df for fp in df.fingerprint.dropna().unique()})
    meta = {"fingerprints": fps, "mixed_hardware": len(fps) > 1}
    if len(fps) > 1:
        msg = f"results come from {len(fps)} different hardware fingerprints: {fps}"
        if not allow_mixed:
            raise SystemExit(f"ERROR: {msg}. Re-run on one machine or pass --allow-mixed-hardware.")
        print(f"WARNING: {msg} (--allow-mixed-hardware: tables will be marked as mixed)")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-mixed-hardware", action="store_true")
    args = ap.parse_args()
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    runs, cost, tuning, frozen = load_runs(), load_runs("cost/raw"), load_tuning(), load_frozen()
    meta = hardware_check([runs, cost, tuning], args.allow_mixed_hardware)
    cost.to_csv(AGG_DIR / "runs_cost.csv", index=False)
    meta["n_cost_runs"] = int(len(cost))
    if len(runs):
        runs = merge_cost(runs, cost)
    tuning.to_csv(AGG_DIR / "tuning.csv", index=False)
    frozen.to_csv(AGG_DIR / "frozen.csv", index=False)
    print(f"tuning runs: {len(tuning)}  frozen configs: {len(frozen)}")
    outs = ("runs.csv", "summary.csv", "summary_tokenizers.csv", "confusion_n8000.csv")
    if runs.empty:
        print("no final runs yet (results/raw is empty)")
        for f in outs:
            (AGG_DIR / f).unlink(missing_ok=True)
        write_json(meta, AGG_DIR / "meta.json")
        return
    runs = runs.sort_values(["dataset", "method", "tokenizer", "size", "seed"])
    runs.to_csv(AGG_DIR / "runs.csv", index=False)
    main_runs = runs[runs.tokenizer == main_tokenizer()]
    summary = summarize(main_runs) if len(main_runs) else pd.DataFrame()
    summary.to_csv(AGG_DIR / "summary.csv", index=False)
    summarize(runs).to_csv(AGG_DIR / "summary_tokenizers.csv", index=False)
    confusion_8000(main_runs).to_csv(AGG_DIR / "confusion_n8000.csv", index=False)
    meta.update(
        train_devices=sorted(runs[runs.family == "neural"].train_device.dropna().unique().tolist()),  # neural only
        accelerator_devices=sorted(runs.acc_device.dropna().unique().tolist()) if "acc_device" in runs else [],
        mamba_backends=sorted(runs[runs.method == "small_mamba"].mamba_backend.dropna().unique().tolist()),
        precisions=sorted(runs.precision.dropna().unique().tolist()),
        n_threads=sorted(runs.n_threads.dropna().astype(int).unique().tolist()),
        gpu_names=sorted(runs.gpu_name.dropna().unique().tolist()),
        cpu_models=sorted(runs.cpu_model.dropna().unique().tolist()),
        mps_fallback_runs=int(runs.mps_fallback_used.fillna(False).astype(bool).sum()),
        n_main_runs=int(len(main_runs)), n_ablation_runs=int((runs.tokenizer != main_tokenizer()).sum()),
    )
    write_json(meta, AGG_DIR / "meta.json")
    print(f"final runs: {len(runs)} (main {len(main_runs)}, ablation-only {meta['n_ablation_runs']})  "
          f"summary cells: {len(summary)} (cells with <5 seeds: {int((summary.n_seeds < 5).sum()) if len(summary) else 0})")
    if len(meta["mamba_backends"]) > 1:
        print(f"  WARNING: Mamba results use more than one backend: {meta['mamba_backends']}")
    if "fp32" not in meta["precisions"] or len(meta["precisions"]) > 1:
        print(f"  WARNING: precisions found: {meta['precisions']} (reported results must be fp32)")
    if len(summary):
        zero = summary[(summary.n_seeds > 1) & (summary.test_macro_f1_std.fillna(1) == 0)]
        for _, r in zero.iterrows():
            print(f"  note: zero std  {r.dataset} {r.method} n={r['size']} (deterministic model on identical data)")
    print(f"  cost runs: {len(cost)} (joined onto {int(runs.timing_reportable.sum())} quality runs; "
          f"cost metrics of all other cells are empty by design)")


if __name__ == "__main__":
    main()
