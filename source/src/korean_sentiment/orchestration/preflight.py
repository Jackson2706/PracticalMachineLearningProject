from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from korean_sentiment.config import checkpoints_dir, data_dir, logs_dir, results_dir
from korean_sentiment.utils.io import atomic_write_json
from korean_sentiment.utils.runtime import hardware_manifest

REQUIRED_PACKAGES = ("torch", "sklearn", "xgboost", "pandas", "numpy", "scipy", "yaml", "psutil", "joblib", "filelock")


def mamba_smoke() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    try:
        from mamba_ssm import Mamba
    except ImportError as error:
        raise RuntimeError("mamba-ssm cannot be imported") from error
    model = Mamba(d_model=32, d_state=8, d_conv=4, expand=2).cuda()
    values = torch.randn(2, 8, 32, device="cuda")
    output = model(values)
    output.sum().backward()
    if output.shape != values.shape or not torch.isfinite(output).all():
        raise RuntimeError("Mamba CUDA forward/backward smoke failed")
    return {"shape": list(output.shape), "device": str(output.device)}


def preflight(require_cloud_neural: bool = False, require_data: bool = False) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "packages": {name: bool(importlib.util.find_spec(name)) for name in REQUIRED_PACKAGES},
        "hardware": hardware_manifest(),
        "nvidia_smi": None,
    }
    errors = []
    if checks["conda_environment"] != "ML":
        errors.append("commands must run inside the Conda environment named ML")
    if sys.version_info[:2] != (3, 11):
        errors.append(f"Python 3.11 is required; found {sys.version.split()[0]}")
    missing = [name for name, available in checks["packages"].items() if not available]
    if missing:
        errors.append(f"missing packages: {', '.join(missing)}")
    for directory in (data_dir(), results_dir(), logs_dir(), checkpoints_dir()):
        directory.mkdir(parents=True, exist_ok=True)
        if not os.access(directory, os.W_OK):
            errors.append(f"path is not writable: {directory}")
    if require_data and not (data_dir() / "manifests" / "data_manifest.json").exists():
        errors.append("preprocessed data manifest is missing")
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
            capture_output=True,
        )
        checks["nvidia_smi"] = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    if require_cloud_neural:
        try:
            checks["mamba_smoke"] = mamba_smoke()
        except RuntimeError as error:
            errors.append(str(error))
    checks["errors"] = errors
    checks["ok"] = not errors
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    atomic_write_json(results_dir() / "artifacts" / f"hardware_manifest_{timestamp}.json", checks)
    if errors:
        raise RuntimeError("preflight failed: " + "; ".join(errors))
    return checks
