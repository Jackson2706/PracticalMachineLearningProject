from __future__ import annotations

import csv
import json
import os
import socket
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from korean_sentiment.config import (
    DATASETS,
    TFIDF_MODELS,
    bonus_config,
    env_int,
    experiment_config,
    model_configs,
    results_dir,
)
from korean_sentiment.contrastive.diagnostics import representation_diagnostics
from korean_sentiment.data.prepare import load_split, load_subset
from korean_sentiment.evaluation.measure import PeakRSS, artifact_size_mb, measure_pipeline
from korean_sentiment.evaluation.metrics import classification_metrics
from korean_sentiment.models import ModelBlocked, load_pipeline, make_pipeline
from korean_sentiment.orchestration.schema import PLAN_FIELDS, RUN_RECORD_FIELDS
from korean_sentiment.orchestration.state import RunLock, now, run_directory, validate_completed
from korean_sentiment.utils.io import atomic_write_csv, atomic_write_json, atomic_write_text, canonical_hash
from korean_sentiment.utils.runtime import configure_threads, hardware_manifest, select_device


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _empty(value: Any) -> bool:
    return value is None or value == ""


def _number(value: Any, kind=float, default=None):
    return default if _empty(value) else kind(value)


def _model_config(model: str, config_id: str) -> dict[str, Any]:
    try:
        return next(config for config in model_configs(model) if str(config["id"]) == str(config_id))
    except StopIteration as error:
        raise ValueError(f"configuration {config_id!r} not found for {model}") from error


def _selected(filename: str, keys: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    path = results_dir() / "csv" / filename
    return {tuple(row[key] for key in keys): row for row in _read_csv(path)} if path.exists() else {}


def _resolve(row: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    phase = row["phase"]
    if phase == "part1_tuning":
        config = _model_config(row["model"], row["config_id"])
    elif phase == "part1_final":
        chosen = _selected("selected_configs.csv", ("dataset", "model")).get((row["dataset"], row["model"]))
        if not chosen:
            raise RuntimeError(f"selected configuration missing for {row['dataset']}/{row['model']}")
        config = _model_config(row["model"], chosen["config_id"])
    else:
        chosen = _selected("selected_configs.csv", ("dataset", "model")).get((row["target_domain"], "transformer"))
        if not chosen:
            raise RuntimeError(f"selected Transformer configuration missing for {row['target_domain']}")
        config = _model_config("transformer", chosen["config_id"])
    objective = {
        "objective": row["objective"],
        "contrastive_lambda": _number(row.get("contrastive_lambda"), float, 0.0),
        "contrastive_temperature": _number(row.get("contrastive_temperature"), float, 0.07),
        "cross_domain_weight": _number(row.get("cross_domain_weight_alpha"), float, 1.0),
    }
    return config, objective


def _part1_data(row: dict[str, str]):
    dataset, seed, train_size = row["dataset"], int(row["seed"]), int(row["train_size"])
    return load_subset(dataset, seed, train_size), load_split(dataset, "validation"), (
        None if row["phase"] == "part1_tuning" else load_split(dataset, "test")
    )


def _bonus_data(row: dict[str, str]):
    target = row["target_domain"]
    sources = row["source_domains"].split(",")
    seed = int(row["seed"])
    target_size = int(row["target_train_size"])
    source_size = int(row["source_size_per_domain"])
    frames = [load_subset(target, seed, target_size)] + [load_subset(source, seed, source_size) for source in sources]
    domains = np.concatenate([np.full(len(frame), index, dtype=np.int64) for index, frame in enumerate(frames)])
    train = pd.concat(frames, ignore_index=True)
    validation = load_split(target, "validation")
    source_validation = [load_split(source, "validation") for source in sources]
    tests = None if row["phase"] == "bonus_tuning" else [load_split(domain, "test") for domain in [target] + sources]
    return train, validation, domains, sources, source_validation, tests


def _resource_record(hardware: dict[str, Any], peak_host: float, training: dict[str, Any]) -> dict[str, Any]:
    return {
        "peak_host_ram_mb": peak_host,
        "peak_gpu_memory_mb": training.get("peak_gpu_memory_mb"),
        "hardware": hardware,
    }


def _flat_record(
    row: dict[str, str],
    status: str,
    metadata: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    resources: dict[str, Any] | None = None,
    model_metadata: dict[str, Any] | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    metrics = metrics or {}
    resources = resources or {}
    model_metadata = model_metadata or {}
    validation = metrics.get("validation", {})
    training = metrics.get("training", {})
    sampler = training.get("sampler") or {}
    test = metrics.get("test", {})
    latency = metrics.get("latency", {})
    classifier = latency.get("classifier", {})
    end_to_end = latency.get("end_to_end", {})
    bonus = metrics.get("bonus", {})
    hardware = resources.get("hardware", {})
    record = {field: None for field in RUN_RECORD_FIELDS}
    record.update({key: row.get(key) for key in PLAN_FIELDS if key in record})
    record.update(
        {
            "status": status,
            "started_at": metadata.get("started_at"),
            "finished_at": metadata.get("finished_at"),
            "duration_sec": metadata.get("duration_sec"),
            "fit_time_sec": metadata.get("fit_time_sec"),
            "best_epoch": metadata.get("best_epoch"),
            "validation_macro_f1": validation.get("macro_f1"),
            "test_macro_f1": test.get("macro_f1"),
            "test_accuracy": test.get("accuracy"),
            "test_macro_precision": test.get("macro_precision"),
            "test_macro_recall": test.get("macro_recall"),
            "precision_negative": test.get("precision_negative"),
            "recall_negative": test.get("recall_negative"),
            "f1_negative": test.get("f1_negative"),
            "precision_positive": test.get("precision_positive"),
            "recall_positive": test.get("recall_positive"),
            "f1_positive": test.get("f1_positive"),
            "tn": test.get("tn"),
            "fp": test.get("fp"),
            "fn": test.get("fn"),
            "tp": test.get("tp"),
            "classifier_latency_ms_mean": classifier.get("mean"),
            "classifier_latency_ms_p50": classifier.get("median"),
            "classifier_latency_ms_p95": classifier.get("p95"),
            "classifier_latency_ms_std": classifier.get("std"),
            "end_to_end_latency_ms_mean": end_to_end.get("mean"),
            "end_to_end_latency_ms_p50": end_to_end.get("median"),
            "end_to_end_latency_ms_p95": end_to_end.get("p95"),
            "end_to_end_latency_ms_std": end_to_end.get("std"),
            "latency_warmup_count": classifier.get("warmup_count"),
            "latency_timed_repeat_count": classifier.get("timed_repeat_count"),
            "throughput_samples_per_sec": latency.get("throughput_samples_per_sec"),
            "peak_host_ram_mb": resources.get("peak_host_ram_mb"),
            "peak_gpu_memory_mb": resources.get("peak_gpu_memory_mb"),
            "serialized_size_mb": metadata.get("serialized_size_mb"),
            "hardware_id": hardware.get("hardware_id"),
            "hostname": hardware.get("hostname"),
            "os": hardware.get("os"),
            "python": hardware.get("python"),
            "git_commit": hardware.get("git_commit"),
            "dirty_worktree": hardware.get("dirty_worktree"),
            "device": metadata.get("device"),
            "precision": metadata.get("precision"),
            "cpu_threads": metadata.get("cpu_threads"),
            "batch_sampler": training.get("batch_sampler"),
            "effective_batch_size": sampler.get("effective_batch_size"),
            "artifact_directory": metadata.get("artifact_directory"),
            "error_type": type(error).__name__ if error else None,
            "error_message": str(error) if error else None,
            **model_metadata,
            **bonus,
        }
    )
    return record


def _evaluation_sample(frames: list[pd.DataFrame], per_cell: int, seed: int):
    generator = np.random.default_rng(seed)
    sampled = []
    domain_ids = []
    for domain, frame in enumerate(frames):
        for label in (0, 1):
            cell = frame.loc[frame["label"] == label].sort_values("example_id")
            take = min(per_cell, len(cell))
            indices = generator.choice(len(cell), size=take, replace=False)
            sampled.append(cell.iloc[indices])
            domain_ids.extend([domain] * take)
    return pd.concat(sampled, ignore_index=True), np.asarray(domain_ids, dtype=np.int64)


def execute_job(row: dict[str, str], force: bool = False) -> dict[str, Any]:
    directory = run_directory(row["run_id"])
    if not force and validate_completed(directory, row["data_manifest_hash"], row["config_hash"]):
        return {"run_id": row["run_id"], "status": "skipped"}
    lock_seconds = int(experiment_config()["orchestration"]["stale_lock_seconds"])
    try:
        lock = RunLock(directory, lock_seconds)
        if not lock.acquire():
            return {"run_id": row["run_id"], "status": "running"}
    except OSError as error:
        return {"run_id": row["run_id"], "status": "failed", "error": str(error)}
    started_clock = time.perf_counter()
    started_at = now()
    log_path = directory / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
    metadata: dict[str, Any] = {
        "run_id": row["run_id"],
        "started_at": started_at,
        "data_manifest_hash": row["data_manifest_hash"],
        "config_hash": row["config_hash"],
        "exact_command": " ".join(os.sys.argv),
    }
    atomic_write_json(directory / "status.json", {"status": "running", "started_at": started_at})
    (directory / "SUCCESS").unlink(missing_ok=True)
    try:
        if row.get("planning_state") != "ready":
            raise RuntimeError(f"job is not executable: {row.get('planning_state')}; regenerate the plan after tuning")
        config, objective = _resolve(row)
        config_without_id = {key: value for key, value in config.items() if key != "id"}
        resolved = {"plan": row, "model": config, "objective": objective}
        atomic_write_json(directory / "resolved_config.json", resolved)
        cpu_threads = configure_threads()
        requested = "cpu" if row["model_family"] == "tfidf" else None
        device = select_device(requested)
        hardware = hardware_manifest()
        metadata.update(
            {
                "device": device,
                "precision": os.environ.get("KS_PRECISION", "fp32"),
                "cpu_threads": cpu_threads,
                "hardware_id": hardware["hardware_id"],
                "artifact_directory": str(directory),
            }
        )
        pipeline = make_pipeline(row["model"], config_without_id, int(row["seed"]), device, cpu_threads)
        with PeakRSS() as memory:
            fit_started = time.perf_counter()
            if row["phase"].startswith("part1"):
                train, validation, test = _part1_data(row)
                training = pipeline.fit(
                    train["text"].tolist(),
                    train["label"].to_numpy(),
                    validation["text"].tolist(),
                    validation["label"].to_numpy(),
                )
                source_validation = []
                tests = None if test is None else [test]
                sources = []
            else:
                train, validation, domains, sources, source_validation, tests = _bonus_data(row)
                training = pipeline.fit(
                    train["text"].tolist(),
                    train["label"].to_numpy(),
                    validation["text"].tolist(),
                    validation["label"].to_numpy(),
                    domains=domains,
                    objective=objective["objective"],
                    contrastive_lambda=objective["contrastive_lambda"],
                    contrastive_temperature=objective["contrastive_temperature"],
                    cross_domain_weight=objective["cross_domain_weight"],
                    target_domain_id=0,
                    effective_batch_size=int(bonus_config()["effective_batch_size"]),
                    example_ids=train["example_id"].tolist(),
                )
            metadata["fit_time_sec"] = time.perf_counter() - fit_started
            metadata["best_epoch"] = training.get("best_epoch")
            validation_metrics = classification_metrics(
                validation["label"], pipeline.predict(validation["text"].tolist())
            )
            metrics: dict[str, Any] = {"validation": validation_metrics, "training": training}
            if source_validation:
                source_validation_scores = [
                    classification_metrics(frame["label"], pipeline.predict(frame["text"].tolist()))["macro_f1"]
                    for frame in source_validation
                ]
                metrics["source_validation_macro_f1"] = source_validation_scores
                metrics["mean_source_validation_macro_f1"] = float(np.mean(source_validation_scores))
            if tests:
                test_metrics = [
                    classification_metrics(frame["label"], pipeline.predict(frame["text"].tolist())) for frame in tests
                ]
                metrics["test"] = test_metrics[0]
                if row["phase"].startswith("bonus"):
                    source_scores = [value["macro_f1"] for value in test_metrics[1:]]
                    evaluation_frame, evaluation_domains = _evaluation_sample(
                        tests,
                        int(experiment_config()["artifacts"]["representation_sample_per_cell"]),
                        int(experiment_config()["artifacts"]["representation_seed"]),
                    )
                    representations = pipeline.representations(evaluation_frame["text"].tolist())
                    diagnostics = representation_diagnostics(
                        representations,
                        evaluation_frame["label"].to_numpy(),
                        evaluation_domains,
                        int(experiment_config()["artifacts"]["representation_pair_count"]),
                        int(experiment_config()["artifacts"]["representation_seed"]),
                    )
                    metrics["bonus"] = {
                        "target_test_macro_f1": test_metrics[0]["macro_f1"],
                        "source_1_test_macro_f1": source_scores[0],
                        "source_2_test_macro_f1": source_scores[1],
                        "mean_source_test_macro_f1": float(np.mean(source_scores)),
                        "all_domain_mean_macro_f1": float(np.mean([value["macro_f1"] for value in test_metrics])),
                        **diagnostics,
                    }
                    if bool(experiment_config()["artifacts"]["retain_embeddings"]):
                        np.savez_compressed(
                            directory / "embeddings.npz",
                            embeddings=representations,
                            labels=evaluation_frame["label"].to_numpy(),
                            domains=evaluation_domains,
                            example_ids=evaluation_frame["example_id"].to_numpy(),
                        )
            latency_settings = experiment_config()["neural"]
            latency_frame = tests[0] if tests else validation
            metrics["latency"] = measure_pipeline(
                pipeline,
                latency_frame["text"].head(32).tolist(),
                int(latency_settings["latency_warmup"]),
                int(latency_settings["latency_repeats"]),
                int(latency_settings["throughput_batch_size"]),
            )
        model_metadata = pipeline.metadata()
        retain = bool(experiment_config()["artifacts"]["retain_models"]) and row["phase"].endswith("final")
        if retain:
            artifact = pipeline.save(directory / "model")
            metadata["serialized_size_mb"] = artifact_size_mb(artifact)
            reloaded = load_pipeline(artifact, device)
            check_texts = validation["text"].head(8).tolist()
            if not np.array_equal(pipeline.predict(check_texts), reloaded.predict(check_texts)):
                raise RuntimeError("serialization reload verification changed predictions")
        else:
            with tempfile.TemporaryDirectory(prefix="ks_model_") as temporary:
                artifact = pipeline.save(Path(temporary) / "model")
                metadata["serialized_size_mb"] = artifact_size_mb(artifact)
                reloaded = load_pipeline(artifact, device)
                check_texts = validation["text"].head(8).tolist()
                if not np.array_equal(pipeline.predict(check_texts), reloaded.predict(check_texts)):
                    raise RuntimeError("serialization reload verification changed predictions")
        metadata.update(finished_at=now(), duration_sec=time.perf_counter() - started_clock)
        resources = _resource_record(hardware, memory.megabytes, training)
        confusion = metrics.get("test", metrics["validation"])["confusion_matrix"]
        atomic_write_json(directory / "run_metadata.json", metadata)
        atomic_write_json(directory / "metrics.json", metrics)
        atomic_write_json(directory / "resources.json", resources)
        atomic_write_json(directory / "confusion_matrix.json", confusion)
        record = _flat_record(row, "completed", metadata, metrics, resources, model_metadata)
        atomic_write_json(directory / "record.json", record)
        atomic_write_json(directory / "status.json", {"status": "completed", "finished_at": metadata["finished_at"]})
        atomic_write_text(directory / "SUCCESS", "validated\n")
        if not validate_completed(directory, row["data_manifest_hash"], row["config_hash"]):
            raise RuntimeError("completed artifact validation failed")
        atomic_write_text(log_path, "completed\n")
        return {"run_id": row["run_id"], "status": "completed"}
    except BaseException as error:
        status = "blocked" if isinstance(error, ModelBlocked) else "failed"
        metadata.update(finished_at=now(), duration_sec=time.perf_counter() - started_clock)
        atomic_write_json(directory / "run_metadata.json", metadata)
        atomic_write_json(
            directory / "error.json",
            {"error_type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
        )
        atomic_write_json(directory / "status.json", {"status": status, "finished_at": metadata["finished_at"]})
        record = _flat_record(row, status, metadata, error=error)
        atomic_write_json(directory / "record.json", record)
        atomic_write_text(log_path, traceback.format_exc())
        return {"run_id": row["run_id"], "status": status, "error": str(error)}
    finally:
        lock.release()


def _matches(row: dict[str, str], filters: dict[str, Any]) -> bool:
    aliases = {"target_size": "target_train_size", "size": "train_size"}
    for key, wanted in filters.items():
        if wanted is None:
            continue
        key = aliases.get(key, key)
        if str(row.get(key, "")) != str(wanted):
            return False
    return True


def execute_plan(
    path: Path,
    filters: dict[str, Any] | None = None,
    maximum_jobs: int | None = None,
    parallelism: int = 1,
    force: bool = False,
    fail_fast: bool = False,
) -> list[dict[str, Any]]:
    rows = [row for row in _read_csv(path) if _matches(row, filters or {})]
    if maximum_jobs is not None:
        rows = rows[:maximum_jobs]
    if parallelism <= 1:
        results = []
        for row in rows:
            result = execute_job(row, force)
            results.append(result)
            if fail_fast and result["status"] in {"failed", "blocked"}:
                break
        return results
    results = []
    with ProcessPoolExecutor(max_workers=parallelism) as pool:
        future_rows = {pool.submit(execute_job, row, force): row for row in rows}
        for future in as_completed(future_rows):
            result = future.result()
            results.append(result)
            if fail_fast and result["status"] in {"failed", "blocked"}:
                for pending in future_rows:
                    pending.cancel()
                break
    return results


def select_part1_configs(require_complete: bool = True) -> list[dict[str, Any]]:
    plan = _read_csv(results_dir() / "plans" / "part1_tuning_plan.csv")
    rows = []
    for item in plan:
        record = run_directory(item["run_id"]) / "record.json"
        if record.exists():
            value = json.loads(record.read_text(encoding="utf-8"))
            if value.get("status") == "completed":
                rows.append(value)
    selected = []
    for dataset in DATASETS:
        for model in experiment_config()["models"]:
            candidates = [row for row in rows if row["dataset"] == dataset and row["model"] == model]
            if require_complete and len(candidates) != 10:
                raise RuntimeError(f"{dataset}/{model}: expected 10 completed tuning trials, found {len(candidates)}")
            if not candidates:
                continue
            winner = sorted(
                candidates,
                key=lambda row: (
                    -float(row["validation_macro_f1"]),
                    float(row["classifier_latency_ms_p50"] or float("inf")),
                    float(row["serialized_size_mb"] or float("inf")),
                    row["config_id"],
                ),
            )[0]
            selected.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "config_id": winner["config_id"],
                    "config_hash": canonical_hash(_model_config(model, winner["config_id"])),
                    "selected_from_run_id": winner["run_id"],
                    "validation_macro_f1": winner["validation_macro_f1"],
                    "validation_inference_latency_ms": winner["classifier_latency_ms_p50"],
                    "serialized_size_mb": winner["serialized_size_mb"],
                }
            )
    atomic_write_csv(
        results_dir() / "csv" / "selected_configs.csv",
        selected,
        ["dataset", "model", "config_id", "config_hash", "selected_from_run_id", "validation_macro_f1", "validation_inference_latency_ms", "serialized_size_mb"],
    )
    return selected


def select_bonus_configs(require_complete: bool = True) -> list[dict[str, Any]]:
    plan = _read_csv(results_dir() / "plans" / "bonus_tuning_plan.csv")
    records = []
    for item in plan:
        path = run_directory(item["run_id"]) / "record.json"
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("status") == "completed":
                metrics = json.loads((path.parent / "metrics.json").read_text(encoding="utf-8"))
                value["mean_source_validation_macro_f1"] = metrics.get("mean_source_validation_macro_f1")
                records.append(value)
    selected = []
    expected = {"supcon": 4, "db_cdsupcon": 8}
    for target in DATASETS:
        for objective in expected:
            candidates = [
                row for row in records if row["target_domain"] == target and row["objective"] == objective
            ]
            if require_complete and len(candidates) != expected[objective]:
                raise RuntimeError(f"{target}/{objective}: expected {expected[objective]} trials, found {len(candidates)}")
            if not candidates:
                continue
            winner = sorted(
                candidates,
                key=lambda row: (
                    -float(row["validation_macro_f1"]),
                    -float(row["mean_source_validation_macro_f1"]),
                    float(row["fit_time_sec"]),
                    row["config_id"],
                ),
            )[0]
            selected.append(
                {
                    "target_domain": target,
                    "objective": objective,
                    "config_id": winner["config_id"],
                    "config_hash": winner["config_hash"],
                    "contrastive_lambda": winner["contrastive_lambda"],
                    "contrastive_temperature": winner["contrastive_temperature"],
                    "cross_domain_weight_alpha": winner["cross_domain_weight_alpha"],
                    "selected_from_run_id": winner["run_id"],
                    "validation_macro_f1": winner["validation_macro_f1"],
                    "mean_source_validation_macro_f1": winner["mean_source_validation_macro_f1"],
                    "fit_time_sec": winner["fit_time_sec"],
                }
            )
    atomic_write_csv(
        results_dir() / "csv" / "bonus_selected_configs.csv",
        selected,
        ["target_domain", "objective", "config_id", "config_hash", "contrastive_lambda", "contrastive_temperature", "cross_domain_weight_alpha", "selected_from_run_id", "validation_macro_f1", "mean_source_validation_macro_f1", "fit_time_sec"],
    )
    return selected
