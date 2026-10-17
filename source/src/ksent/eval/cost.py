"""Cost measurement: wall-clock timers, batch-1 latency, peak host memory, artefact size.

Accelerator synchronisation and memory come from :mod:`ksent.hardware`.
"""
from __future__ import annotations

import os
import resource
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ksent.hardware import synchronize


class Timer:
    """Context manager measuring wall-clock seconds, synchronising ``device`` on entry/exit."""

    def __init__(self, device: str | None = None) -> None:
        self.device, self.seconds = device, 0.0

    def __enter__(self) -> "Timer":
        if self.device:
            synchronize(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        if self.device:
            synchronize(self.device)
        self.seconds = time.perf_counter() - self._t0


def ru_maxrss_mb() -> float:
    """Peak resident set size of this process (macOS reports bytes, Linux KiB)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10


class RSSSampler:
    """Background thread sampling this process' RSS with psutil (default every 10 ms)."""

    def __init__(self, interval_s: float = 0.01) -> None:
        import psutil

        self._proc, self.interval, self.peak_bytes = psutil.Process(os.getpid()), interval_s, 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self._proc.memory_info().rss)
            self._stop.wait(self.interval)

    def start(self) -> "RSSSampler":
        self._thread.start()
        return self

    def current_mb(self) -> float:
        """Current RSS in MB."""
        return self._proc.memory_info().rss / 2**20

    def stop(self) -> float:
        """Stop sampling and return the peak in MB."""
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self._proc.memory_info().rss)
        return self.peak_bytes / 2**20


def measure_latency(fn: Callable[[Any], Any], inputs: Sequence[Any], n_warmup: int, device: str = "cpu") -> dict:
    """Time ``fn(x)`` once per input (batch size 1) after ``n_warmup`` warm-up calls.

    Every call is bracketed by ``synchronize(device)``. Returns median / p95 / mean in ms.
    """
    for i in range(n_warmup):
        fn(inputs[i % len(inputs)])
    synchronize(device)
    times = []
    for x in inputs:
        synchronize(device)
        t0 = time.perf_counter()
        fn(x)
        synchronize(device)
        times.append((time.perf_counter() - t0) * 1e3)
    a = np.asarray(times)
    return {"median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(a.mean()), "n": int(len(a)), "n_warmup": int(n_warmup), "device": device}


def file_size_mb(p: str | Path) -> float:
    """Size of a file (or the sum over a directory) in MB."""
    p = Path(p)
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2**20
    return p.stat().st_size / 2**20
