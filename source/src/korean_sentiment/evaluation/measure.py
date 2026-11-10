from __future__ import annotations

import statistics
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


def latency_summary(values_ms: list[float], warmup: int, repeats: int) -> dict[str, float | int]:
    return {
        "mean": float(statistics.fmean(values_ms)),
        "median": float(statistics.median(values_ms)),
        "p95": float(np.percentile(values_ms, 95)),
        "std": float(statistics.pstdev(values_ms)),
        "warmup_count": warmup,
        "timed_repeat_count": repeats,
    }


def measure_call(function: Callable[[], Any], warmup: int, repeats: int, device: str) -> dict[str, float | int]:
    import torch

    for _ in range(warmup):
        function()
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    values = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        function()
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        values.append((time.perf_counter_ns() - started) / 1_000_000)
    return latency_summary(values, warmup, repeats)


def measure_pipeline(pipeline, texts: list[str], warmup: int, repeats: int, throughput_batch_size: int) -> dict[str, Any]:
    sample = texts[0]
    prepared = pipeline.prepare([sample])
    classifier = measure_call(lambda: pipeline.predict_prepared(prepared), warmup, repeats, pipeline.device)
    end_to_end = measure_call(lambda: pipeline.predict([sample]), warmup, repeats, pipeline.device)
    batch = (texts * (throughput_batch_size // max(1, len(texts)) + 1))[:throughput_batch_size]
    started = time.perf_counter()
    pipeline.predict(batch)
    elapsed = time.perf_counter() - started
    return {
        "classifier": classifier,
        "end_to_end": end_to_end,
        "throughput_batch_size": throughput_batch_size,
        "throughput_samples_per_sec": throughput_batch_size / max(elapsed, 1e-12),
    }


class PeakRSS:
    def __init__(self, interval: float = 0.02) -> None:
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self) -> "PeakRSS":
        import psutil

        process = psutil.Process()

        def sample() -> None:
            while not self._stop.wait(self.interval):
                self.peak = max(self.peak, process.memory_info().rss)

        self.peak = process.memory_info().rss
        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=1)

    @property
    def megabytes(self) -> float:
        return self.peak / 1024**2


def artifact_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1024**2
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / 1024**2
