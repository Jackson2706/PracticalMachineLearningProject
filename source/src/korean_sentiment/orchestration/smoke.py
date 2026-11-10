from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from korean_sentiment.config import MODELS, model_configs, results_dir
from korean_sentiment.evaluation.metrics import classification_metrics
from korean_sentiment.models import ModelBlocked, load_pipeline, make_pipeline
from korean_sentiment.orchestration.state import run_directory
from korean_sentiment.utils.io import atomic_write_csv, atomic_write_json, atomic_write_text
from korean_sentiment.utils.runtime import configure_threads, hardware_manifest, select_device


def _fixture():
    positive = ["정말 재미있고 좋아요", "배송이 빠르고 만족합니다", "훌륭한 작품 :) ", "추천합니다 최고예요"]
    negative = ["너무 지루하고 싫어요", "배송이 느리고 불만입니다", "별로인 작품 :( ", "추천하지 않아요"]
    train = (positive + negative) * 3
    labels = np.asarray(([1] * 4 + [0] * 4) * 3)
    validation = positive[:2] + negative[:2]
    validation_labels = np.asarray([1, 1, 0, 0])
    return train, labels, validation, validation_labels


def _small_config(model: str) -> dict[str, Any]:
    config = {key: value for key, value in model_configs(model)[0].items() if key != "id"}
    if model == "tfidf_rf":
        config["n_estimators"] = 5
    if model == "tfidf_xgb":
        config["n_estimators"] = 5
    return config


def _write_result(name: str, status: str, metrics: dict | None, error: BaseException | None = None) -> None:
    directory = run_directory(f"smoke-{name}")
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    metadata = {"run_id": f"smoke-{name}", "phase": "smoke", "name": name, "hardware": hardware_manifest()}
    atomic_write_json(directory / "resolved_config.json", {"name": name})
    atomic_write_json(directory / "run_metadata.json", metadata)
    atomic_write_json(directory / "status.json", {"status": status})
    atomic_write_json(directory / "metrics.json", metrics or {})
    atomic_write_json(directory / "resources.json", {})
    atomic_write_json(directory / "confusion_matrix.json", (metrics or {}).get("confusion_matrix", []))
    record = {
        "run_id": f"smoke-{name}",
        "phase": "smoke",
        "status": status,
        "model": name,
        "error_type": type(error).__name__ if error else None,
        "error_message": str(error) if error else None,
    }
    atomic_write_json(directory / "record.json", record)
    if error:
        atomic_write_json(directory / "error.json", {"error_type": type(error).__name__, "message": str(error)})
        atomic_write_text(directory / f"run_{timestamp}.log", f"{type(error).__name__}: {error}\n")
    else:
        atomic_write_text(directory / f"run_{timestamp}.log", "completed\n")
    if status == "completed":
        atomic_write_text(directory / "SUCCESS", "validated\n")


def run_smoke(device_request: str | None = None) -> list[dict[str, Any]]:
    train, labels, validation, validation_labels = _fixture()
    threads = configure_threads()
    device = select_device(device_request, allow_fallback=True)
    rows = []
    for model in MODELS:
        started = time.perf_counter()
        pipeline = None
        try:
            selected_device = "cpu" if model.startswith("tfidf_") else device
            pipeline = make_pipeline(model, _small_config(model), 13, selected_device, threads)
            fit = pipeline.fit(
                train,
                labels,
                validation,
                validation_labels,
                max_epochs=1,
                patience=1,
            )
            predictions = pipeline.predict(validation)
            metrics = classification_metrics(validation_labels, predictions)
            with tempfile.TemporaryDirectory(prefix="ks_smoke_") as temporary:
                artifact = pipeline.save(Path(temporary) / model)
                loaded = load_pipeline(artifact, selected_device)
                if not np.array_equal(predictions, loaded.predict(validation)):
                    raise RuntimeError("reload predictions differ")
            _write_result(model, "completed", metrics)
            rows.append({"pipeline": model, "status": "completed", "duration_sec": time.perf_counter() - started})
        except ModelBlocked as error:
            _write_result(model, "blocked", None, error)
            rows.append({"pipeline": model, "status": "blocked_on_mac", "duration_sec": time.perf_counter() - started, "reason": str(error)})
        except BaseException as error:
            _write_result(model, "failed", None, error)
            rows.append({"pipeline": model, "status": "failed", "duration_sec": time.perf_counter() - started, "reason": str(error)})
    pooled_train = []
    pooled_labels = []
    domains = []
    for domain in range(3):
        for label, text in ((0, "정말 별로예요"), (1, "정말 좋아요")):
            for index in range(4):
                pooled_train.append(f"{text} {domain} {index}")
                pooled_labels.append(label)
                domains.append(domain)
    for objective in ("pooled_ce", "supcon", "db_cdsupcon"):
        started = time.perf_counter()
        try:
            pipeline = make_pipeline("transformer", _small_config("transformer"), 13, device, threads)
            pipeline.fit(
                pooled_train,
                np.asarray(pooled_labels),
                validation,
                validation_labels,
                domains=np.asarray(domains),
                objective=objective,
                contrastive_lambda=0.1,
                contrastive_temperature=0.07,
                cross_domain_weight=2.0,
                target_domain_id=0,
                effective_batch_size=12,
                max_epochs=1,
                patience=1,
            )
            predictions = pipeline.predict(validation)
            metrics = classification_metrics(validation_labels, predictions)
            with tempfile.TemporaryDirectory(prefix="ks_bonus_smoke_") as temporary:
                artifact = pipeline.save(Path(temporary) / objective)
                loaded = load_pipeline(artifact, device)
                if not np.array_equal(predictions, loaded.predict(validation)):
                    raise RuntimeError("reload predictions differ")
            _write_result(objective, "completed", metrics)
            rows.append({"pipeline": objective, "status": "completed", "duration_sec": time.perf_counter() - started})
        except BaseException as error:
            _write_result(objective, "failed", None, error)
            rows.append({"pipeline": objective, "status": "failed", "duration_sec": time.perf_counter() - started, "reason": str(error)})
    fields = ["pipeline", "status", "duration_sec", "reason"]
    atomic_write_csv(results_dir() / "csv" / "smoke_status.csv", rows, fields)
    return rows
