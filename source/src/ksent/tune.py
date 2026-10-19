"""Select the tuned configuration per (dataset, method) and freeze it.

Selection: highest validation macro-F1 among the 10 predefined configs at
n=2,000 / tuning seed 0; ties broken by lower end-to-end fit time. For XGBoost the
early-stopped best round count is frozen as ``n_estimators``.

Usage: ``python -m ksent.tune`` (after the cost-mode tuning stage)

Quality route (exclusive tuning estimated > 8 h): ``--source tuning_quality`` selects from
the parallel runs in ``results/tuning_quality/tuning/``; after the 270 fits are replayed in
cost mode, ``--source tuning_quality --times-from tuning`` keeps that selection, takes every
reported tuning time from ``results/tuning/`` and checks that the replay reproduces each
config's validation macro-F1.
"""
from __future__ import annotations

import argparse

from ksent.config import dump_yaml, frozen_path, global_cfg, path, resolve_config
from ksent.utils import now_iso, read_json


def _root(source: str):
    return path("results") / "tuning" if source == "tuning" else path("results") / source / "tuning"


def _load(dataset: str, method: str, source: str) -> list[dict]:
    d = _root(source) / dataset / method
    runs = []
    for k in range(10):
        p = d / f"cfg{k}.json"
        if not p.exists():
            raise FileNotFoundError(f"missing tuning result {p}")
        r = read_json(p)
        if r.get("status") != "ok":
            raise RuntimeError(f"tuning run failed: {p}")
        runs.append(r)
    return runs


def select(dataset: str, method: str, source: str = "tuning", times_from: str | None = None) -> dict:
    """Pick the winning config of ``(dataset, method)``; raise if tuning is incomplete."""
    runs = _load(dataset, method, source)
    timed = runs
    if times_from:
        timed = _load(dataset, method, times_from)
        for a, b in zip(runs, timed):
            if abs(a["val"]["macro_f1"] - b["val"]["macro_f1"]) > 1e-9 and method.startswith("tfidf"):
                raise RuntimeError(f"{dataset}/{method} cfg{a['config_id']}: replay val F1 differs "
                                   f"({a['val']['macro_f1']} vs {b['val']['macro_f1']})")
    best = max(runs, key=lambda r: (round(r["val"]["macro_f1"], 10), -r["cost"]["fit_time_end_to_end_s"]))
    params = resolve_config(method, best["config_id"])
    if method == "tfidf_xgb":
        params["n_estimators"] = int(best["fit"]["xgb_best_rounds"])
    return {
        "dataset": dataset, "method": method, "selected_config_id": best["config_id"], "params": params,
        "val_macro_f1": best["val"]["macro_f1"],
        "tuning_time_selected_s": best["cost"]["fit_time_end_to_end_s"],
        "tuning_time_total_s": sum(r["cost"]["fit_time_end_to_end_s"] for r in timed),
        "tuning_wall_time_total_s": sum(r["wall_time_s"] for r in timed),
        "tuning_times_reportable": all(r.get("timing_reportable", True) for r in timed),
        "selection_source": source, "times_source": times_from or source,
        "replay_val_f1_max_abs_diff": max(abs(a["val"]["macro_f1"] - b["val"]["macro_f1"])
                                          for a, b in zip(runs, timed)),
        "all_configs": [{"config_id": r["config_id"], "val_macro_f1": r["val"]["macro_f1"],
                         "fit_time_s": r["cost"]["fit_time_end_to_end_s"],
                         **({"xgb_best_rounds": r["fit"]["xgb_best_rounds"]} if method == "tfidf_xgb" else {})}
                        for r in runs],
        "tuning_size": global_cfg()["tuning_size"], "tuning_seed": global_cfg()["tuning_seed"],
        "frozen_at": now_iso(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", nargs="*", default=None)
    ap.add_argument("--method", nargs="*", default=None)
    ap.add_argument("--source", default="tuning", help="tuning (cost runs) | tuning_quality")
    ap.add_argument("--times-from", default=None, help="take tuning times from this source (cost replay)")
    args = ap.parse_args()
    g = global_cfg()
    grand = 0.0
    for ds in args.dataset or g["datasets"]:
        for m in args.method or g["methods"]:
            sel = select(ds, m, args.source, args.times_from)
            dump_yaml(sel, frozen_path(ds, m))
            grand += sel["tuning_time_total_s"]
            print(f"[{ds:8s}] {m:18s} cfg{sel['selected_config_id']} val_f1={sel['val_macro_f1']:.4f} "
                  f"tuning={sel['tuning_time_total_s']:.1f}s  {sel['params']}")
    print(f"total tuning fit time: {grand:.1f}s")


if __name__ == "__main__":
    main()
