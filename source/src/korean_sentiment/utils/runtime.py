from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from korean_sentiment.config import PROJECT_ROOT, env_int
from korean_sentiment.utils.io import canonical_hash


def seed_everything(seed: int, deterministic: bool = True) -> list[str]:
    random.seed(seed)
    np.random.seed(seed)
    notes: list[str] = []
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
        if torch.backends.mps.is_available():
            notes.append("PyTorch documents that some MPS operations may remain nondeterministic")
    except ImportError:
        notes.append("PyTorch unavailable")
    return notes


def configure_threads() -> int:
    count = env_int("KS_CPU_THREADS", max(1, min(4, os.cpu_count() or 1)))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(count)
    try:
        import torch

        torch.set_num_threads(count)
    except (ImportError, RuntimeError):
        pass
    return count


def select_device(requested: str | None = None, allow_fallback: bool = False) -> str:
    requested = (requested or os.environ.get("KS_DEVICE") or "auto").lower()
    import torch

    def mps_usable() -> bool:
        if not torch.backends.mps.is_available():
            return False
        try:
            torch.ones(1, device="mps").cpu()
            return True
        except RuntimeError:
            return False

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if mps_usable():
            return "mps"
        return "cpu"
    available = requested == "cpu" or (requested.startswith("cuda") and torch.cuda.is_available()) or (
        requested == "mps" and mps_usable()
    )
    if available:
        return requested
    if allow_fallback:
        return "cpu"
    raise RuntimeError(f"requested device {requested!r} is unavailable")


def git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True
            ).stdout.strip()
        )
        return {"git_commit": commit, "dirty_worktree": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "dirty_worktree": None}


def hardware_manifest() -> dict[str, Any]:
    try:
        import psutil

        ram = psutil.virtual_memory().total
        disk = psutil.disk_usage(PROJECT_ROOT).free
    except ImportError:
        ram = disk = None
    gpu: dict[str, Any] = {"available": False}
    try:
        import torch

        if torch.cuda.is_available():
            gpu = {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "count": torch.cuda.device_count(),
                "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
                "torch_cuda": torch.version.cuda,
            }
        mps = bool(torch.backends.mps.is_available())
        if mps:
            try:
                torch.ones(1, device="mps").cpu()
            except RuntimeError:
                mps = False
    except ImportError:
        mps = False
    packages = {}
    for name in ("torch", "numpy", "pandas", "scikit-learn", "xgboost", "mamba-ssm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    identity = {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "ram_bytes": ram,
        "disk_free_bytes": disk,
        "gpu": gpu,
        "mps_available": mps,
        "packages": packages,
        **git_state(),
    }
    identity["hardware_id"] = canonical_hash(
        {key: identity[key] for key in ("hostname", "os", "machine", "processor", "cpu_count", "ram_bytes", "gpu")}
    )[:16]
    return identity
