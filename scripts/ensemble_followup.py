"""Follow-up experiment: probability ensemble of TF-IDF and pooled neural models.

For every (target domain, target size, seed) cell of the bonus grid this script
re-trains, with the frozen configurations and the exact plan rows used for the
audited results:

  * the best TF-IDF model of Part 1 at that target size (target labels only), and
  * the pooled neural models (pooled CE and/or DB-CDSupCon) of the bonus,

saves their per-review probabilities on the target validation and test sets, and
evaluates the ensemble

    p_ens = w * p_tfidf + (1 - w) * p_neural

with two weight rules:
  * ``avg``  : w = 0.5, fixed in advance (primary result, nothing tuned), and
  * ``valw`` : w chosen on target validation macro-F1 from {0.0, 0.1, ..., 1.0}.

LinearSVC has no probabilities, so its decision scores are Platt-calibrated on the
target validation set (the same set neural early stopping already uses). The test
set is used only for the final numbers.

Nothing in the audited results is changed: outputs go to source/results/followup_ensemble/.

Usage (from the repository root, inside the ML conda env, on the GPU machine;
the processed data must exist: ./run_all.sh download && ./run_all.sh preprocess):

    python scripts/ensemble_followup.py run --device cuda --targets steam --sizes 200 --seeds 13   # one cell
    python scripts/ensemble_followup.py run --device cuda                                          # all 36 cells
    python scripts/ensemble_followup.py aggregate

Outputs: source/results/followup_ensemble/

``run`` skips cells whose outputs already exist (use --force to redo them).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _package_root() -> Path:
    """Find source/src whether this file sits in <repo>/scripts or <repo>/source/scripts."""
    for parent in Path(__file__).resolve().parents:
        for candidate in (parent / "source" / "src", parent / "src"):
            if (candidate / "korean_sentiment").is_dir():
                return candidate
    raise SystemExit("cannot find source/src/korean_sentiment; run from inside the repository")


sys.path.insert(0, str(_package_root()))

from korean_sentiment.config import bonus_config, results_dir  # noqa: E402
from korean_sentiment.data.prepare import load_split, load_subset  # noqa: E402
from korean_sentiment.models import make_pipeline  # noqa: E402
from korean_sentiment.orchestration.executor import _bonus_data, _resolve  # noqa: E402
from korean_sentiment.utils.runtime import configure_threads, select_device  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

TARGETS = ["nsmc", "shopping", "steam"]
SIZES = [200, 500, 1000, 2000]
SEEDS = [13, 42, 2026]
OBJECTIVES = ["pooled_ce", "db_cdsupcon"]
WEIGHTS = [round(0.1 * i, 1) for i in range(11)]


def out_dir() -> Path:
    path = results_dir() / "followup_ensemble"
    (path / "cells").mkdir(parents=True, exist_ok=True)
    return path


def read_plan(name: str) -> list[dict[str, str]]:
    with (results_dir() / "csv" / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def best_tfidf_model(target: str, size: int) -> str:
    """Best TF-IDF model at this size by the audited three-seed mean (same rule as the slides)."""
    path = results_dir() / "reporting" / "derived_metrics" / "part1_best_by_size.csv"
    table = pd.read_csv(path)
    row = table[(table.dataset == target) & (table.train_size == size)]
    if row.empty:
        raise RuntimeError(f"no Part 1 best-by-size row for {target}/{size}")
    return str(row.iloc[0].best_tfidf_model)


def find_row(plan: list[dict[str, str]], **wanted: Any) -> dict[str, str]:
    rows = [r for r in plan if all(str(r[k]) == str(v) for k, v in wanted.items())]
    if len(rows) != 1:
        raise RuntimeError(f"expected one plan row for {wanted}, found {len(rows)}")
    return rows[0]


def original_score(run_id: str, key: str) -> float | None:
    path = results_dir() / "run_records" / run_id / "record.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8")).get(key)
    return None if value in (None, "") else float(value)


def macro_f1(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(f1_score(labels, (probabilities >= 0.5).astype(int), average="macro", zero_division=0))


# ---------------------------------------------------------------- TF-IDF
def tfidf_probabilities(target: str, size: int, seed: int) -> dict[str, Any]:
    model = best_tfidf_model(target, size)
    row = find_row(read_plan("experiment_plan.csv"), phase="part1_final", dataset=target,
                   model=model, train_size=size, seed=seed)
    config, _ = _resolve(row)
    config = {k: v for k, v in config.items() if k != "id"}
    pipeline = make_pipeline(model, config, seed, "cpu", configure_threads())
    train = load_subset(target, seed, size)
    validation, test = load_split(target, "validation"), load_split(target, "test")
    started = time.perf_counter()
    pipeline.fit(train["text"].tolist(), train["label"].to_numpy())
    fit_time = time.perf_counter() - started
    val_scores = pipeline.decision_scores(validation["text"].tolist())
    test_scores = pipeline.decision_scores(test["text"].tolist())
    if hasattr(pipeline.estimator, "predict_proba"):
        p_val, p_test, calibration = val_scores, test_scores, "native_predict_proba"
    else:  # LinearSVC: Platt scaling on target validation
        platt = LogisticRegression(C=1e6, max_iter=1000)
        platt.fit(val_scores.reshape(-1, 1), validation["label"].to_numpy())
        p_val = platt.predict_proba(val_scores.reshape(-1, 1))[:, 1]
        p_test = platt.predict_proba(test_scores.reshape(-1, 1))[:, 1]
        calibration = "platt_on_target_validation"
    # hard predictions of the uncalibrated model, to check we reproduce Part 1 exactly
    reproduced = float(f1_score(test["label"], pipeline.predict(test["text"].tolist()), average="macro"))
    return {
        "model": model, "run_id": row["run_id"], "calibration": calibration, "fit_time_sec": fit_time,
        "p_val": p_val, "p_test": p_test,
        "reproduced_test_macro_f1": reproduced,
        "original_test_macro_f1": original_score(row["run_id"], "test_macro_f1"),
    }


# ---------------------------------------------------------------- neural
def neural_probabilities(pipeline, texts: list[str], batch_size: int = 256) -> np.ndarray:
    import torch

    ids, lengths = pipeline.prepare(texts)
    pipeline.model.eval()
    out = []
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            logits = pipeline.model(ids[start:start + batch_size], lengths[start:start + batch_size])
            out.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def pooled_probabilities(target: str, size: int, seed: int, objective: str, device: str,
                         max_epochs: int | None) -> dict[str, Any]:
    row = find_row(read_plan("bonus_experiment_plan.csv"), phase="bonus_final", target_domain=target,
                   target_train_size=size, seed=seed, objective=objective)
    config, obj = _resolve(row)
    config = {k: v for k, v in config.items() if k != "id"}
    pipeline = make_pipeline("transformer", config, seed, device, configure_threads())
    train, validation, domains, _, _, tests = _bonus_data(row)
    started = time.perf_counter()
    pipeline.fit(
        train["text"].tolist(), train["label"].to_numpy(),
        validation["text"].tolist(), validation["label"].to_numpy(),
        domains=domains, objective=obj["objective"],
        contrastive_lambda=obj["contrastive_lambda"],
        contrastive_temperature=obj["contrastive_temperature"],
        cross_domain_weight=obj["cross_domain_weight"],
        target_domain_id=0,
        effective_batch_size=int(bonus_config()["effective_batch_size"]),
        example_ids=train["example_id"].tolist(),
        **({"max_epochs": max_epochs} if max_epochs else {}),
    )
    fit_time = time.perf_counter() - started
    test = tests[0]
    p_val = neural_probabilities(pipeline, validation["text"].tolist())
    p_test = neural_probabilities(pipeline, test["text"].tolist())
    return {
        "objective": objective, "run_id": row["run_id"], "fit_time_sec": fit_time,
        "p_val": p_val, "p_test": p_test,
        "reproduced_test_macro_f1": macro_f1(test["label"].to_numpy(), p_test),
        "original_test_macro_f1": original_score(row["run_id"], "target_test_macro_f1"),
    }


# ---------------------------------------------------------------- one cell
def run_cell(target: str, size: int, seed: int, objectives: list[str], device: str,
             force: bool, max_epochs: int | None) -> None:
    cell = out_dir() / "cells" / f"{target}_{size}_{seed}"
    if (cell / "SUCCESS").exists() and not force:
        print(f"skip {cell.name}")
        return
    cell.mkdir(parents=True, exist_ok=True)
    validation, test = load_split(target, "validation"), load_split(target, "test")
    y_val, y_test = validation["label"].to_numpy(), test["label"].to_numpy()

    tfidf = tfidf_probabilities(target, size, seed)
    arrays = {"val_example_id": validation["example_id"].to_numpy(), "val_label": y_val,
              "test_example_id": test["example_id"].to_numpy(), "test_label": y_test,
              "val_p_tfidf": tfidf["p_val"], "test_p_tfidf": tfidf["p_test"]}
    rows: list[dict[str, Any]] = []
    base = {"target_domain": target, "target_train_size": size, "seed": seed}
    rows.append({**base, "method": "best_tfidf", "component": tfidf["model"], "weight_tfidf": 1.0,
                 "test_macro_f1": macro_f1(y_test, tfidf["p_test"]),
                 "test_accuracy": float(accuracy_score(y_test, tfidf["p_test"] >= 0.5)),
                 "reproduced_hard_test_macro_f1": tfidf["reproduced_test_macro_f1"],
                 "original_test_macro_f1": tfidf["original_test_macro_f1"],
                 "fit_time_sec": tfidf["fit_time_sec"], "run_id": tfidf["run_id"],
                 "note": tfidf["calibration"]})
    for objective in objectives:
        neural = pooled_probabilities(target, size, seed, objective, device, max_epochs)
        arrays[f"val_p_{objective}"] = neural["p_val"]
        arrays[f"test_p_{objective}"] = neural["p_test"]
        rows.append({**base, "method": objective, "component": "transformer", "weight_tfidf": 0.0,
                     "test_macro_f1": macro_f1(y_test, neural["p_test"]),
                     "test_accuracy": float(accuracy_score(y_test, neural["p_test"] >= 0.5)),
                     "reproduced_hard_test_macro_f1": neural["reproduced_test_macro_f1"],
                     "original_test_macro_f1": neural["original_test_macro_f1"],
                     "fit_time_sec": neural["fit_time_sec"], "run_id": neural["run_id"], "note": ""})
        # ensemble, fixed equal weights (primary)
        p = 0.5 * tfidf["p_test"] + 0.5 * neural["p_test"]
        rows.append({**base, "method": f"ens_avg_{objective}", "component": f"{tfidf['model']}+{objective}",
                     "weight_tfidf": 0.5, "test_macro_f1": macro_f1(y_test, p),
                     "test_accuracy": float(accuracy_score(y_test, p >= 0.5)),
                     "fit_time_sec": tfidf["fit_time_sec"] + neural["fit_time_sec"], "note": "w fixed at 0.5"})
        # ensemble, weight selected on target validation (secondary)
        val_scores = [macro_f1(y_val, w * tfidf["p_val"] + (1 - w) * neural["p_val"]) for w in WEIGHTS]
        w = WEIGHTS[int(np.argmax(val_scores))]
        p = w * tfidf["p_test"] + (1 - w) * neural["p_test"]
        rows.append({**base, "method": f"ens_valw_{objective}", "component": f"{tfidf['model']}+{objective}",
                     "weight_tfidf": w, "test_macro_f1": macro_f1(y_test, p),
                     "test_accuracy": float(accuracy_score(y_test, p >= 0.5)),
                     "fit_time_sec": tfidf["fit_time_sec"] + neural["fit_time_sec"],
                     "note": f"w chosen on validation macro-F1 ({max(val_scores):.4f})"})
    np.savez_compressed(cell / "probabilities.npz", **arrays)
    pd.DataFrame(rows).to_csv(cell / "metrics.csv", index=False)
    (cell / "SUCCESS").write_text("ok\n", encoding="utf-8")
    print(pd.DataFrame(rows)[["method", "weight_tfidf", "test_macro_f1", "original_test_macro_f1"]].to_string(index=False))


# ---------------------------------------------------------------- aggregate
def aggregate() -> None:
    frames = [pd.read_csv(p) for p in sorted((out_dir() / "cells").glob("*/metrics.csv"))]
    if not frames:
        raise SystemExit("no finished cells")
    runs = pd.concat(frames, ignore_index=True)
    runs.to_csv(out_dir() / "ensemble_metrics_by_seed.csv", index=False)
    summary = (runs.groupby(["target_domain", "target_train_size", "method"])
               .agg(test_macro_f1_mean=("test_macro_f1", "mean"), test_macro_f1_sd=("test_macro_f1", "std"),
                    weight_tfidf_mean=("weight_tfidf", "mean"), fit_time_sec_mean=("fit_time_sec", "mean"),
                    n_seeds=("seed", "nunique"))
               .reset_index())
    summary.to_csv(out_dir() / "ensemble_summary.csv", index=False)
    check = runs.dropna(subset=["original_test_macro_f1"])
    if not check.empty:
        check = check.assign(abs_diff=(check.test_macro_f1 - check.original_test_macro_f1).abs())
        tfidf = check[check.method == "best_tfidf"].assign(
            abs_diff=(check.reproduced_hard_test_macro_f1 - check.original_test_macro_f1).abs())
        print(f"reproduction check: TF-IDF max |diff| = {tfidf.abs_diff.max():.4f}; "
              f"neural max |diff| = {check[check.method != 'best_tfidf'].abs_diff.max():.4f}")
    wide = summary.pivot_table(index=["target_domain", "target_train_size"], columns="method",
                               values="test_macro_f1_mean").round(3)
    print(wide.to_string())
    wide.to_csv(out_dir() / "ensemble_summary_wide.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--targets", nargs="+", default=TARGETS)
    run.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    run.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    run.add_argument("--objectives", nargs="+", default=OBJECTIVES)
    run.add_argument("--device", default=None, help="cuda, cpu or mps (default: auto)")
    run.add_argument("--force", action="store_true")
    run.add_argument("--max-epochs", type=int, default=None, help="smoke testing only")
    sub.add_parser("aggregate")
    args = parser.parse_args()
    if args.command == "aggregate":
        aggregate()
        return
    from korean_sentiment.config import data_dir
    missing = [d for d in TARGETS if not (data_dir() / "processed" / d / "test.csv").exists()]
    if missing:
        raise SystemExit(f"processed data missing for {missing}: run ./run_all.sh download && ./run_all.sh preprocess")
    device = select_device(args.device)
    for target in args.targets:
        for size in args.sizes:
            for seed in args.seeds:
                print(f"=== {target} size={size} seed={seed} device={device}", flush=True)
                run_cell(target, size, seed, args.objectives, device, args.force, args.max_epochs)
    aggregate()


if __name__ == "__main__":
    main()