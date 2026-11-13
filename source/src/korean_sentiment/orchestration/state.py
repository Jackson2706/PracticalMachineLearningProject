from __future__ import annotations

import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from korean_sentiment.config import results_dir
from korean_sentiment.utils.io import atomic_write_json, read_json

REQUIRED_COMPLETED = (
    "record.json",
    "resolved_config.json",
    "run_metadata.json",
    "status.json",
    "metrics.json",
    "resources.json",
    "confusion_matrix.json",
    "SUCCESS",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_directory(run_id: str) -> Path:
    return results_dir() / "run_records" / run_id


def validate_completed(path: Path, data_hash: str | None = None, config_hash: str | None = None) -> bool:
    if not all((path / name).exists() for name in REQUIRED_COMPLETED):
        return False
    try:
        status = read_json(path / "status.json")
        metadata = read_json(path / "run_metadata.json")
        metrics = read_json(path / "metrics.json")
        record = read_json(path / "record.json")
    except (OSError, json.JSONDecodeError):
        return False
    if status.get("status") != "completed" or not isinstance(metrics, dict):
        return False
    if record.get("status") != "completed" or record.get("run_id") != path.name:
        return False
    if data_hash is not None and metadata.get("data_manifest_hash") != data_hash:
        atomic_write_json(path / "status.json", {**status, "status": "stale", "stale_at": now()})
        return False
    if config_hash is not None and metadata.get("config_hash") != config_hash:
        atomic_write_json(path / "status.json", {**status, "status": "stale", "stale_at": now()})
        return False
    return True


class RunLock:
    def __init__(self, directory: Path, stale_seconds: int = 21600) -> None:
        self.directory = directory
        self.path = directory / ".lock"
        self.stale_seconds = stale_seconds
        self.acquired = False

    def _stale(self) -> bool:
        try:
            value = read_json(self.path)
        except (OSError, json.JSONDecodeError):
            return time.time() - self.path.stat().st_mtime > self.stale_seconds
        age = time.time() - float(value.get("epoch", 0))
        if value.get("hostname") == socket.gethostname():
            try:
                os.kill(int(value["pid"]), 0)
                return False
            except (OSError, KeyError, ValueError):
                return True
        return age > self.stale_seconds

    def acquire(self) -> bool:
        self.directory.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump({"pid": os.getpid(), "hostname": socket.gethostname(), "epoch": time.time()}, handle)
                self.acquired = True
                return True
            except FileExistsError:
                if not self._stale():
                    return False
                self.path.unlink(missing_ok=True)
                status_path = self.directory / "status.json"
                if status_path.exists():
                    status = read_json(status_path)
                    atomic_write_json(status_path, {**status, "status": "interrupted", "interrupted_at": now()})
        return False

    def release(self) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False

    def __enter__(self) -> "RunLock":
        if not self.acquire():
            raise BlockingIOError(f"run is locked: {self.directory.name}")
        return self

    def __exit__(self, *_: Any) -> None:
        self.release()
