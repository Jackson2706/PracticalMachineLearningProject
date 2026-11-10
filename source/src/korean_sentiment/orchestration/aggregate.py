from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from korean_sentiment.config import DATASETS, FINAL_SEEDS, TFIDF_MODELS, bonus_config, results_dir
from korean_sentiment.contrastive.diagnostics import normalized_log_auc
from korean_sentiment.orchestration.schema import PLAN_FIELDS, RUN_RECORD_FIELDS, STATUSES
from korean_sentiment.orchestration.state import run_directory
from korean_sentiment.utils.io import atomic_write_csv


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def collect_records() -> list[dict[str, Any]]:
    records = []
    for path in sorted((results_dir() / "run_records").glob("*/record.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        value["artifact_directory"] = str(path.parent)
        records.append(value)
    return sorted(records, key=lambda row: row.get("run_id", ""))


def _write(name: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    atomic_write_csv(results_dir() / "csv" / name, rows, fields)


def _status_rows(plan: list[dict[str, str]], records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for job in plan:
        record = records.get(job["run_id"])
        status = record.get("status") if record else "pending"
        if status not in STATUSES:
            status = "failed"
        rows.append({**job, "status": status, "error_type": record.get("error_type") if record else None, "error_message": record.get("error_message") if record else None})
    return rows


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _part1_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("phase") == "part1_final" and row.get("status") == "completed":
            groups[(row["dataset"], row["model"], int(row["train_size"]))].append(row)
    output = []
    for (dataset, model, size), values in sorted(groups.items()):
        scores = [float(value["test_macro_f1"]) for value in values]
        mean, std = _mean_std(scores)
        output.append(
            {
                "dataset": dataset,
                "model": model,
                "model_family": values[0]["model_family"],
                "train_size": size,
                "test_macro_f1_mean": mean,
                "test_macro_f1_std": std,
                "completed_seeds": len({int(value["seed"]) for value in values}),
                "expected_seeds": len(FINAL_SEEDS),
                "complete": len({int(value["seed"]) for value in values}) == len(FINAL_SEEDS),
                "run_ids": ",".join(sorted(value["run_id"] for value in values)),
            }
        )
    by_curve: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        by_curve[(row["dataset"], row["model"])].append(row)
    for curve in by_curve.values():
        curve.sort(key=lambda row: row["train_size"])
        for index, row in enumerate(curve):
            previous = curve[index - 1] if index else None
            row["gain_from_previous_size"] = None if previous is None else row["test_macro_f1_mean"] - previous["test_macro_f1_mean"]
            candidates = [prior for prior in curve[:index] if prior["train_size"] <= row["train_size"] / 2]
            doubled = candidates[-1] if candidates else None
            row["approximately_doubled_from_size"] = doubled["train_size"] if doubled else None
            row["gain_from_approximately_half_size"] = None if doubled is None else row["test_macro_f1_mean"] - doubled["test_macro_f1_mean"]
    return output


def _crossovers(summary: list[dict[str, Any]], plan: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    datasets = sorted({row["dataset"] for row in plan if row.get("phase") == "part1_final"})
    for dataset in datasets:
        sizes = sorted({int(row["train_size"]) for row in plan if row.get("phase") == "part1_final" and row["dataset"] == dataset})
        size_rows = []
        for size in sizes:
            candidates = [row for row in summary if row["dataset"] == dataset and row["train_size"] == size]
            tfidf = [row for row in candidates if row["model_family"] == "tfidf" and row["complete"]]
            neural = [row for row in candidates if row["model_family"] == "neural" and row["complete"]]
            best_tfidf = max(tfidf, key=lambda row: row["test_macro_f1_mean"], default=None)
            best_neural = max(neural, key=lambda row: row["test_macro_f1_mean"], default=None)
            difference = None if not best_tfidf or not best_neural else best_neural["test_macro_f1_mean"] - best_tfidf["test_macro_f1_mean"]
            size_rows.append(
                {
                    "dataset": dataset,
                    "training_size": size,
                    "best_tfidf_model": best_tfidf["model"] if best_tfidf else None,
                    "best_tfidf_mean_macro_f1": best_tfidf["test_macro_f1_mean"] if best_tfidf else None,
                    "best_neural_model": best_neural["model"] if best_neural else None,
                    "best_neural_mean_macro_f1": best_neural["test_macro_f1_mean"] if best_neural else None,
                    "difference": difference,
                    "criterion_satisfied": difference is not None and difference >= 0.01,
                    "missing_run_warning": None if best_tfidf and best_neural else "one or both model families lack a complete three-seed result",
                }
            )
        observed = [row["training_size"] for row in size_rows if row["criterion_satisfied"]]
        sustained = [
            row["training_size"]
            for index, row in enumerate(size_rows)
            if row["criterion_satisfied"] and all(later["criterion_satisfied"] for later in size_rows[index:])
        ]
        for row in size_rows:
            row["first_observed_crossover"] = min(observed) if observed else None
            row["first_sustained_crossover"] = min(sustained) if sustained else None
            row["sustained_definition"] = "criterion is satisfied at this size and every larger tested size"
        output.extend(size_rows)
    return output


def _bonus_rows(records: list[dict[str, Any]], part1: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    completed_part1 = [row for row in part1 if row.get("phase") == "part1_final" and row.get("status") == "completed"]
    for target in DATASETS:
        for size in bonus_config()["target_sizes"]:
            for seed in FINAL_SEEDS:
                base = [row for row in completed_part1 if row["dataset"] == target and int(row["train_size"]) == size and int(row["seed"]) == seed]
                tfidf = max((row for row in base if row["model"] in TFIDF_MODELS), key=lambda row: float(row["test_macro_f1"]), default=None)
                transformer = next((row for row in base if row["model"] == "transformer"), None)
                for method, value in (("best_tfidf", tfidf), ("target_transformer", transformer)):
                    if value:
                        output.append(
                            {
                                "target_domain": target,
                                "target_train_size": size,
                                "seed": seed,
                                "method": method,
                                "objective": value["objective"],
                                "target_test_macro_f1": value["test_macro_f1"],
                                "target_accuracy": value.get("test_accuracy"),
                                "target_macro_precision": value.get("test_macro_precision"),
                                "target_macro_recall": value.get("test_macro_recall"),
                                "mean_source_test_macro_f1": None,
                                "all_domain_mean_macro_f1": None,
                                "record_origin": "reused_part1",
                                "run_id": value["run_id"],
                                "reused_run_ids": value["run_id"],
                                "complete": True,
                            }
                        )
    for row in records:
        if row.get("phase") == "bonus_final" and row.get("status") == "completed":
            output.append(
                {
                    "target_domain": row["target_domain"],
                    "target_train_size": int(row["target_train_size"]),
                    "seed": int(row["seed"]),
                    "method": row["objective"],
                    "objective": row["objective"],
                    "target_test_macro_f1": row["target_test_macro_f1"],
                    "target_accuracy": row.get("test_accuracy"),
                    "target_macro_precision": row.get("test_macro_precision"),
                    "target_macro_recall": row.get("test_macro_recall"),
                    "mean_source_test_macro_f1": row["mean_source_test_macro_f1"],
                    "all_domain_mean_macro_f1": row["all_domain_mean_macro_f1"],
                    "record_origin": "new_bonus",
                    "run_id": row["run_id"],
                    "reused_run_ids": None,
                    "complete": True,
                }
            )
    return sorted(output, key=lambda row: (row["target_domain"], row["method"], row["target_train_size"], row["seed"]))


def _bonus_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["target_domain"], row["method"], row["target_train_size"])].append(row)
    output = []
    for (target, method, size), values in sorted(groups.items()):
        scores = [float(value["target_test_macro_f1"]) for value in values]
        mean, std = _mean_std(scores)
        source = [float(value["mean_source_test_macro_f1"]) for value in values if value["mean_source_test_macro_f1"] not in (None, "")]
        output.append(
            {
                "target_domain": target,
                "method": method,
                "target_train_size": size,
                "target_macro_f1_mean": mean,
                "target_macro_f1_std": std,
                "mean_source_macro_f1_mean": float(np.mean(source)) if source else None,
                "completed_seeds": len({int(value["seed"]) for value in values}),
                "expected_seeds": len(FINAL_SEEDS),
                "complete": len({int(value["seed"]) for value in values}) == len(FINAL_SEEDS),
                "run_ids": ",".join(sorted(value["run_id"] for value in values)),
                "record_origins": ",".join(sorted({value["record_origin"] for value in values})),
            }
        )
    pooled = {(row["target_domain"], row["target_train_size"]): row for row in output if row["method"] == "pooled_ce"}
    for row in output:
        reference = pooled.get((row["target_domain"], row["target_train_size"]))
        row["source_retention_difference_vs_pooled_ce"] = (
            None
            if not reference or row["mean_source_macro_f1_mean"] is None or reference["mean_source_macro_f1_mean"] is None
            else row["mean_source_macro_f1_mean"] - reference["mean_source_macro_f1_mean"]
        )
    return output


def _bonus_efficiency(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for target in sorted({row["target_domain"] for row in summary}):
        tfidf = {row["target_train_size"]: row["target_macro_f1_mean"] for row in summary if row["target_domain"] == target and row["method"] == "best_tfidf" and row["complete"]}
        thresholds: dict[str, int | None] = {}
        for method in sorted({row["method"] for row in summary if row["target_domain"] == target}):
            curve = sorted((row for row in summary if row["target_domain"] == target and row["method"] == method and row["complete"]), key=lambda row: row["target_train_size"])
            thresholds[method] = next((row["target_train_size"] for row in curve if row["target_train_size"] in tfidf and row["target_macro_f1_mean"] >= tfidf[row["target_train_size"]] + 0.01), None)
            auc = normalized_log_auc([row["target_train_size"] for row in curve], [row["target_macro_f1_mean"] for row in curve]) if len(curve) >= 2 else None
            output.append(
                {
                    "target_domain": target,
                    "method": method,
                    "crossover_threshold_size": thresholds[method],
                    "threshold_status": "observed" if thresholds[method] is not None else "not_observed_within_tested_sizes",
                    "normalized_log2_learning_curve_auc": auc,
                    "auc_rule": "trapezoidal integral over log2(target size), divided by the tested log2 range",
                }
            )
        reference = thresholds.get("target_transformer")
        for row in output:
            if row["target_domain"] == target:
                candidate = row["crossover_threshold_size"]
                row["target_only_threshold_size"] = reference
                row["target_label_crossover_shift"] = None if reference is None or candidate is None else candidate - reference
                row["target_training_labels_saved"] = None if reference is None or candidate is None else max(0, reference - candidate)
                row["savings_interpretation"] = "new target-domain labels saved when labeled source-domain data already exists"
    return output


def aggregate_all() -> dict[str, int]:
    records = collect_records()
    _write("run_records.csv", records, RUN_RECORD_FIELDS)
    tuning = [row for row in records if row.get("phase") == "part1_tuning"]
    _write("tuning_trials.csv", tuning, RUN_RECORD_FIELDS)
    part1_seed = [row for row in records if row.get("phase") == "part1_final"]
    _write("final_metrics_by_seed.csv", part1_seed, RUN_RECORD_FIELDS)
    summary = _part1_summary(records)
    summary_fields = [
        "dataset", "model", "model_family", "train_size", "test_macro_f1_mean", "test_macro_f1_std",
        "completed_seeds", "expected_seeds", "complete", "gain_from_previous_size",
        "approximately_doubled_from_size", "gain_from_approximately_half_size", "run_ids",
    ]
    _write("final_metrics_summary.csv", summary, summary_fields)
    part1_plan = [row for row in _read(results_dir() / "csv" / "experiment_plan.csv") if row["phase"].startswith("part1")]
    crossovers = _crossovers(summary, part1_plan)
    _write("crossover_summary.csv", crossovers, list(crossovers[0]) if crossovers else ["dataset", "training_size"])
    resources = [row for row in records if row.get("status") == "completed"]
    _write("resource_summary.csv", resources, RUN_RECORD_FIELDS)
    record_map = {row["run_id"]: row for row in records}
    part1_status = _status_rows(part1_plan, record_map)
    _write("status.csv", part1_status, PLAN_FIELDS + ["status", "error_type", "error_message"])
    bonus_tuning = [row for row in records if row.get("phase") == "bonus_tuning"]
    _write("bonus_tuning_trials.csv", bonus_tuning, RUN_RECORD_FIELDS)
    bonus_seed = _bonus_rows(records, records)
    bonus_seed_fields = [
        "target_domain", "target_train_size", "seed", "method", "objective", "target_test_macro_f1",
        "target_accuracy", "target_macro_precision", "target_macro_recall",
        "mean_source_test_macro_f1", "all_domain_mean_macro_f1", "record_origin", "run_id", "reused_run_ids", "complete",
    ]
    _write("bonus_metrics_by_seed.csv", bonus_seed, bonus_seed_fields)
    bonus_summary = _bonus_summary(bonus_seed)
    bonus_summary_fields = [
        "target_domain", "method", "target_train_size", "target_macro_f1_mean", "target_macro_f1_std",
        "mean_source_macro_f1_mean", "source_retention_difference_vs_pooled_ce", "completed_seeds",
        "expected_seeds", "complete", "run_ids", "record_origins",
    ]
    _write("bonus_metrics_summary.csv", bonus_summary, bonus_summary_fields)
    efficiency = _bonus_efficiency(bonus_summary)
    efficiency_fields = list(efficiency[0]) if efficiency else ["target_domain", "method"]
    _write("bonus_label_efficiency.csv", efficiency, efficiency_fields)
    representation = [row for row in records if row.get("phase") == "bonus_final" and row.get("status") == "completed"]
    _write("bonus_representation_metrics.csv", representation, RUN_RECORD_FIELDS)
    bonus_plan = [row for row in _read(results_dir() / "csv" / "bonus_experiment_plan.csv") if row["phase"].startswith("bonus")]
    bonus_status = _status_rows(bonus_plan, record_map)
    _write("bonus_status.csv", bonus_status, PLAN_FIELDS + ["status", "error_type", "error_message"])
    complete_status = part1_status + bonus_status
    _write("complete_project_status.csv", complete_status, PLAN_FIELDS + ["status", "error_type", "error_message"])
    selected_fields = ["dataset", "model", "config_id", "config_hash", "selected_from_run_id", "validation_macro_f1", "validation_inference_latency_ms", "serialized_size_mb"]
    if not (results_dir() / "csv" / "selected_configs.csv").exists():
        _write("selected_configs.csv", [], selected_fields)
    bonus_selected_fields = ["target_domain", "objective", "config_id", "config_hash", "contrastive_lambda", "contrastive_temperature", "cross_domain_weight_alpha", "selected_from_run_id", "validation_macro_f1", "mean_source_validation_macro_f1", "fit_time_sec"]
    if not (results_dir() / "csv" / "bonus_selected_configs.csv").exists():
        _write("bonus_selected_configs.csv", [], bonus_selected_fields)
    return {
        "run_records": len(records),
        "part1_final_records": len(part1_seed),
        "bonus_new_final_records": len([row for row in records if row.get("phase") == "bonus_final"]),
    }
