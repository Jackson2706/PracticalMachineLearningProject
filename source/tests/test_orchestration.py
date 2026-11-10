from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import pytest

from korean_sentiment.config import results_dir
from korean_sentiment.orchestration.aggregate import aggregate_all
from korean_sentiment.orchestration.executor import execute_job
from korean_sentiment.orchestration.planner import build_bonus_plans, build_part1_plans, stable_run_id
from korean_sentiment.orchestration.schema import PLAN_FIELDS, RUN_RECORD_FIELDS
from korean_sentiment.orchestration.state import RunLock, run_directory, validate_completed
from korean_sentiment.utils.io import atomic_write_json, atomic_write_text, read_json


def test_stable_run_id_is_order_independent_and_sensitive():
    value = {
        "phase": "part1_final",
        "dataset": "nsmc",
        "model": "transformer",
        "objective": "ce",
        "train_size": 200,
        "seed": 13,
        "config_id": "tr01",
        "data_manifest_hash": "data",
        "config_hash": "config",
    }
    assert stable_run_id(value) == stable_run_id(dict(reversed(list(value.items()))))
    assert stable_run_id(value) != stable_run_id({**value, "seed": 42})


def test_atomic_writes_replace_complete_files(tmp_path: Path):
    path = tmp_path / "value.json"
    atomic_write_json(path, {"a": 1})
    atomic_write_json(path, {"a": 2})
    assert read_json(path) == {"a": 2}
    assert not list(tmp_path.glob("*.tmp"))


def test_lock_prevents_duplicate_and_recovers_stale(tmp_path: Path):
    directory = tmp_path / "run"
    first = RunLock(directory, stale_seconds=1)
    assert first.acquire()
    second = RunLock(directory, stale_seconds=1)
    assert not second.acquire()
    first.release()
    atomic_write_json(directory / ".lock", {"pid": 99999999, "hostname": os.uname().nodename, "epoch": 0})
    recovered = RunLock(directory, stale_seconds=1)
    assert recovered.acquire()
    recovered.release()


def test_completed_validation_and_stale_hash(isolated_paths: Path):
    directory = run_directory("complete")
    directory.mkdir(parents=True)
    for name in ("resolved_config.json", "metrics.json", "resources.json", "confusion_matrix.json"):
        atomic_write_json(directory / name, {})
    atomic_write_json(directory / "run_metadata.json", {"data_manifest_hash": "a", "config_hash": "b"})
    atomic_write_json(directory / "status.json", {"status": "completed"})
    atomic_write_text(directory / "SUCCESS", "ok\n")
    assert validate_completed(directory, "a", "b")
    assert not validate_completed(directory, "changed", "b")
    assert read_json(directory / "status.json")["status"] == "stale"


def test_planner_expected_counts(isolated_paths: Path, monkeypatch: pytest.MonkeyPatch):
    import korean_sentiment.orchestration.planner as planner

    monkeypatch.setattr(planner, "_statistics", lambda: {"nsmc": 135000, "shopping": 160000, "steam": 80000})
    monkeypatch.setattr(planner, "data_manifest_hash", lambda: "manifest")
    part1 = build_part1_plans()
    assert part1["tuning_jobs"] == 270
    assert part1["final_jobs"] == 837
    assert part1["total_jobs"] == 1107
    bonus = build_bonus_plans()
    assert bonus["tuning_jobs"] == 36
    assert bonus["final_jobs"] == 108
    assert bonus["new_jobs"] == 144
    assert bonus["reused_reference_rows"] == 72
    with (results_dir() / "csv" / "bonus_experiment_plan.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 144


def test_pending_plan_fails_atomically_with_record(isolated_paths: Path):
    row = {field: "" for field in PLAN_FIELDS}
    row.update(
        {
            "run_id": "pending-job",
            "phase": "part1_final",
            "dataset": "nsmc",
            "model": "tfidf_lr",
            "model_family": "tfidf",
            "objective": "ce",
            "train_size": "200",
            "seed": "13",
            "config_id": "selected_pending",
            "data_manifest_hash": "data",
            "config_hash": "config",
            "planning_state": "awaiting_selected_config",
        }
    )
    result = execute_job(row)
    assert result["status"] == "failed"
    directory = run_directory("pending-job")
    assert read_json(directory / "status.json")["status"] == "failed"
    assert (directory / "error.json").exists()
    assert not (directory / "SUCCESS").exists()


def test_aggregation_is_deterministic_and_emits_required_csvs(isolated_paths: Path):
    record = {field: None for field in RUN_RECORD_FIELDS}
    record.update(
        {
            "run_id": "one",
            "phase": "part1_final",
            "status": "completed",
            "dataset": "nsmc",
            "model": "tfidf_lr",
            "model_family": "tfidf",
            "objective": "ce",
            "train_size": 200,
            "seed": 13,
            "test_macro_f1": 0.7,
        }
    )
    directory = run_directory("one")
    directory.mkdir(parents=True)
    atomic_write_json(directory / "record.json", record)
    first = aggregate_all()
    content = (results_dir() / "csv" / "run_records.csv").read_text(encoding="utf-8")
    second = aggregate_all()
    assert first == second
    assert content == (results_dir() / "csv" / "run_records.csv").read_text(encoding="utf-8")
    required = {
        "tuning_trials.csv", "run_records.csv", "final_metrics_by_seed.csv", "final_metrics_summary.csv",
        "crossover_summary.csv", "resource_summary.csv", "status.csv", "bonus_tuning_trials.csv",
        "bonus_metrics_by_seed.csv", "bonus_metrics_summary.csv", "bonus_label_efficiency.csv",
        "bonus_representation_metrics.csv", "bonus_status.csv", "complete_project_status.csv",
    }
    assert required <= {path.name for path in (results_dir() / "csv").glob("*.csv")}
