from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any

from korean_sentiment.config import (
    DATASETS,
    FINAL_SEEDS,
    MODELS,
    NEURAL_MODELS,
    TFIDF_MODELS,
    TUNING_SEED,
    bonus_config,
    data_dir,
    experiment_config,
    model_configs,
    results_dir,
)
from korean_sentiment.data.prepare import data_manifest_hash, size_schedule
from korean_sentiment.orchestration.schema import PLAN_FIELDS
from korean_sentiment.utils.io import atomic_write_csv, atomic_write_json, canonical_hash


def stable_run_id(specification: dict[str, Any]) -> str:
    identity_fields = (
        "phase", "dataset", "target_domain", "source_domains", "model", "objective", "train_size",
        "target_train_size", "source_size_per_domain", "seed", "config_id", "data_manifest_hash", "config_hash",
    )
    identity = {key: specification.get(key) for key in identity_fields}
    prefix = "bonus" if str(specification.get("phase", "")).startswith("bonus") else "part1"
    return f"{prefix}-{canonical_hash(identity)[:20]}"


def _statistics() -> dict[str, int]:
    path = results_dir() / "csv" / "dataset_statistics.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run preprocess first")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = {row["dataset"]: int(row["n_full"]) for row in rows}
    missing = set(DATASETS) - set(values)
    if missing:
        raise ValueError(f"dataset statistics missing: {sorted(missing)}")
    return values


def _selected_configs(filename: str, keys: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, Any]]:
    path = results_dir() / "csv" / filename
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {tuple(row[key] for key in keys): row for row in rows}


def _finish(row: dict[str, Any]) -> dict[str, Any]:
    row = {key: row.get(key) for key in PLAN_FIELDS if key != "run_id"}
    row["run_id"] = stable_run_id(row)
    return row


def build_part1_plans() -> dict[str, Any]:
    sizes = _statistics()
    manifest_hash = data_manifest_hash()
    tuning_rows = []
    for dataset, model in itertools.product(DATASETS, MODELS):
        for config in model_configs(model):
            config_hash = canonical_hash(config)
            tuning_rows.append(
                _finish(
                    {
                        "phase": "part1_tuning",
                        "dataset": dataset,
                        "model": model,
                        "model_family": "tfidf" if model in TFIDF_MODELS else "neural",
                        "objective": "ce",
                        "train_size": int(experiment_config()["tuning_size"]),
                        "train_fraction": int(experiment_config()["tuning_size"]) / sizes[dataset],
                        "seed": TUNING_SEED,
                        "config_id": config["id"],
                        "data_manifest_hash": manifest_hash,
                        "config_hash": config_hash,
                        "planning_state": "ready",
                    }
                )
            )
    selected = _selected_configs("selected_configs.csv", ("dataset", "model"))
    final_rows = []
    schedules = {dataset: size_schedule(n_full) for dataset, n_full in sizes.items()}
    for dataset, model, train_size, seed in (
        (dataset, model, train_size, seed)
        for dataset in DATASETS
        for model in MODELS
        for train_size in schedules[dataset]
        for seed in FINAL_SEEDS
    ):
        chosen = selected.get((dataset, model))
        config_id = chosen["config_id"] if chosen else "selected_pending"
        config_hash = chosen["config_hash"] if chosen else canonical_hash({"dataset": dataset, "model": model, "pending": True})
        final_rows.append(
            _finish(
                {
                    "phase": "part1_final",
                    "dataset": dataset,
                    "model": model,
                    "model_family": "tfidf" if model in TFIDF_MODELS else "neural",
                    "objective": "ce",
                    "train_size": train_size,
                    "train_fraction": train_size / sizes[dataset],
                    "seed": seed,
                    "config_id": config_id,
                    "data_manifest_hash": manifest_hash,
                    "config_hash": config_hash,
                    "planning_state": "ready" if chosen else "awaiting_selected_config",
                }
            )
        )
    plans = results_dir() / "plans"
    csv_dir = results_dir() / "csv"
    atomic_write_csv(plans / "part1_tuning_plan.csv", tuning_rows, PLAN_FIELDS)
    atomic_write_csv(plans / "part1_final_plan.csv", final_rows, PLAN_FIELDS)
    atomic_write_csv(csv_dir / "experiment_plan.csv", tuning_rows + final_rows, PLAN_FIELDS)
    expected_conditions = 31
    actual_conditions = sum(len(values) for values in schedules.values())
    explanation = (
        "matches the expected cleaned-pool schedules"
        if actual_conditions == expected_conditions
        else f"data-derived schedules contain {actual_conditions} conditions instead of {expected_conditions}"
    )
    summary = {
        "dataset_full_sizes": sizes,
        "dataset_schedules": schedules,
        "tuning_jobs": len(tuning_rows),
        "final_jobs": len(final_rows),
        "total_jobs": len(tuning_rows) + len(final_rows),
        "expected_tuning_jobs": 270,
        "expected_final_jobs": 837,
        "difference_explanation": explanation,
    }
    atomic_write_json(plans / "part1_plan_summary.json", summary)
    return summary


def _bonus_configs(objective: str) -> list[dict[str, Any]]:
    config = bonus_config()
    if objective == "supcon":
        return [
            {"id": f"supcon_l{lam}_t{tau}", "lambda": lam, "temperature": tau, "alpha": None}
            for lam, tau in itertools.product(config["standard_supcon"]["lambdas"], config["standard_supcon"]["temperatures"])
        ]
    if objective == "db_cdsupcon":
        return [
            {"id": f"db_l{lam}_t{tau}_a{alpha}", "lambda": lam, "temperature": tau, "alpha": alpha}
            for lam, tau, alpha in itertools.product(
                config["db_cdsupcon"]["lambdas"],
                config["db_cdsupcon"]["temperatures"],
                config["db_cdsupcon"]["alphas"],
            )
        ]
    raise KeyError(objective)


def build_bonus_plans() -> dict[str, Any]:
    _statistics()
    manifest_hash = data_manifest_hash()
    part1_selected = _selected_configs("selected_configs.csv", ("dataset", "model"))
    bonus_selected = _selected_configs("bonus_selected_configs.csv", ("target_domain", "objective"))
    scenarios = {target: sorted(set(DATASETS) - {target}) for target in DATASETS}
    tuning_rows = []
    for target in DATASETS:
        backbone = part1_selected.get((target, "transformer"))
        for objective in ("supcon", "db_cdsupcon"):
            for config in _bonus_configs(objective):
                combined = {"backbone": backbone["config_hash"] if backbone else "pending", "contrastive": config}
                tuning_rows.append(
                    _finish(
                        {
                            "phase": "bonus_tuning",
                            "dataset": target,
                            "target_domain": target,
                            "source_domains": ",".join(scenarios[target]),
                            "model": "transformer",
                            "model_family": "neural",
                            "objective": objective,
                            "train_size": 4500,
                            "target_train_size": 500,
                            "source_size_per_domain": 2000,
                            "seed": TUNING_SEED,
                            "config_id": config["id"],
                            "contrastive_lambda": config["lambda"],
                            "contrastive_temperature": config["temperature"],
                            "cross_domain_weight_alpha": config["alpha"],
                            "data_manifest_hash": manifest_hash,
                            "config_hash": canonical_hash(combined),
                            "planning_state": "ready" if backbone else "awaiting_part1_transformer_config",
                        }
                    )
                )
    final_rows = []
    for target, objective, target_size, seed in itertools.product(
        DATASETS, bonus_config()["objectives"], bonus_config()["target_sizes"], FINAL_SEEDS
    ):
        backbone = part1_selected.get((target, "transformer"))
        contrastive = bonus_selected.get((target, objective)) if objective != "pooled_ce" else None
        ready = bool(backbone) and (objective == "pooled_ce" or bool(contrastive))
        config_id = backbone["config_id"] if objective == "pooled_ce" and backbone else (
            contrastive["config_id"] if contrastive else "selected_pending"
        )
        combined = {
            "backbone": backbone["config_hash"] if backbone else "pending",
            "contrastive": contrastive["config_hash"] if contrastive else objective,
        }
        final_rows.append(
            _finish(
                {
                    "phase": "bonus_final",
                    "dataset": target,
                    "target_domain": target,
                    "source_domains": ",".join(scenarios[target]),
                    "model": "transformer",
                    "model_family": "neural",
                    "objective": objective,
                    "train_size": target_size + 4000,
                    "target_train_size": target_size,
                    "source_size_per_domain": 2000,
                    "seed": seed,
                    "config_id": config_id,
                    "contrastive_lambda": contrastive.get("contrastive_lambda") if contrastive else None,
                    "contrastive_temperature": contrastive.get("contrastive_temperature") if contrastive else None,
                    "cross_domain_weight_alpha": contrastive.get("cross_domain_weight_alpha") if contrastive else None,
                    "data_manifest_hash": manifest_hash,
                    "config_hash": canonical_hash(combined),
                    "planning_state": "ready" if ready else "awaiting_selected_config",
                }
            )
        )
    references = []
    part1_plan_path = results_dir() / "plans" / "part1_final_plan.csv"
    part1_rows = []
    if part1_plan_path.exists():
        with part1_plan_path.open(encoding="utf-8", newline="") as handle:
            part1_rows = list(csv.DictReader(handle))
    for target, size, seed, kind in itertools.product(DATASETS, bonus_config()["target_sizes"], FINAL_SEEDS, ("best_tfidf", "target_transformer")):
        candidates = [
            row for row in part1_rows
            if row["dataset"] == target and int(row["train_size"]) == size and int(row["seed"]) == seed
            and ((kind == "best_tfidf" and row["model"] in TFIDF_MODELS) or (kind == "target_transformer" and row["model"] == "transformer"))
        ]
        references.append(
            {
                "target_domain": target,
                "target_train_size": size,
                "seed": seed,
                "reference_type": kind,
                "candidate_run_ids": ",".join(sorted(row["run_id"] for row in candidates)),
                "resolved_run_id": candidates[0]["run_id"] if kind == "target_transformer" and len(candidates) == 1 else "",
            }
        )
    plans = results_dir() / "plans"
    csv_dir = results_dir() / "csv"
    atomic_write_csv(plans / "bonus_tuning_plan.csv", tuning_rows, PLAN_FIELDS)
    atomic_write_csv(plans / "bonus_final_plan.csv", final_rows, PLAN_FIELDS)
    atomic_write_csv(csv_dir / "bonus_experiment_plan.csv", tuning_rows + final_rows, PLAN_FIELDS)
    atomic_write_csv(
        plans / "bonus_reuse_references.csv",
        references,
        ["target_domain", "target_train_size", "seed", "reference_type", "candidate_run_ids", "resolved_run_id"],
    )
    summary = {
        "tuning_jobs": len(tuning_rows),
        "final_jobs": len(final_rows),
        "new_jobs": len(tuning_rows) + len(final_rows),
        "reused_reference_rows": len(references),
        "expected_tuning_jobs": 36,
        "expected_final_jobs": 108,
        "expected_complete_project_jobs": 1251,
    }
    atomic_write_json(plans / "bonus_plan_summary.json", summary)
    return summary


def validate_bonus_dependencies() -> None:
    selected = _selected_configs("selected_configs.csv", ("dataset", "model"))
    missing_configs = [dataset for dataset in DATASETS if (dataset, "transformer") not in selected]
    if missing_configs:
        raise RuntimeError(f"bonus planning requires frozen Part 1 Transformer configs for: {', '.join(missing_configs)}")
    records_path = results_dir() / "csv" / "final_metrics_by_seed.csv"
    records = _read_csv_rows(records_path) if records_path.exists() else []
    missing_references = []
    for dataset, size, seed in itertools.product(DATASETS, bonus_config()["target_sizes"], FINAL_SEEDS):
        rows = [
            row for row in records
            if row.get("dataset") == dataset and row.get("train_size") == str(size)
            and row.get("seed") == str(seed) and row.get("status") == "completed"
        ]
        if not any(row.get("model") == "transformer" for row in rows) or not any(row.get("model") in TFIDF_MODELS for row in rows):
            missing_references.append(f"{dataset}/n{size}/seed{seed}")
    if missing_references:
        preview = ", ".join(missing_references[:6])
        raise RuntimeError(f"bonus planning requires completed Part 1 Transformer and TF-IDF references ({preview})")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
