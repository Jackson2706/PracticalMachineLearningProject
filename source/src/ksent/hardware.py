"""Hardware abstraction: device resolution, synchronisation, memory, seeding,
determinism, threads and environment information.

This is the ONLY module that touches ``torch.cuda.*`` / ``torch.mps.*``; everything
else goes through these helpers so the code runs unchanged on NVIDIA GPUs (CUDA),
Apple Silicon (MPS) and CPU-only machines.

Torch is imported lazily so that TF-IDF runs never load it (keeps their memory
footprint honest).
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import platform
import random
import shutil
import subprocess
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ksent.config import global_cfg


class DeviceUnavailableError(RuntimeError):
    """Raised when an explicitly requested device is not available."""


# ----------------------------------------------------------------------------- availability
def _torch():
    import torch

    return torch


def cuda_available() -> bool:
    return _torch().cuda.is_available()


def mps_available() -> bool:
    torch = _torch()
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def accelerators() -> list[str]:
    """Available accelerator device strings (at most one CUDA device is used)."""
    out = []
    if cuda_available():
        out.append("cuda")
    if mps_available():
        out.append("mps")
    return out


def configured_device() -> str:
    """Device spec from ``KSENT_DEVICE`` or ``global.yaml: device``."""
    return os.environ.get("KSENT_DEVICE") or str(global_cfg().get("device", "auto"))


def configured_threads() -> int:
    """Thread count from ``KSENT_THREADS`` or ``global.yaml: n_threads``, capped at the core count."""
    n = int(os.environ.get("KSENT_THREADS") or global_cfg().get("n_threads", 8))
    return max(1, min(n, os.cpu_count() or n))


def resolve_device(spec: str | None = None, allow_fallback: bool = False) -> tuple[str, str]:
    """Resolve a device spec to a concrete device and a human-readable reason.

    ``auto`` picks CUDA -> MPS -> CPU. An explicitly requested but unavailable device
    raises :class:`DeviceUnavailableError` unless ``allow_fallback`` (smoke tests only),
    in which case the ``auto`` choice is used and the reason says so.
    """
    spec = (spec or configured_device()).strip().lower()
    if spec == "auto":
        if cuda_available():
            return "cuda", "auto: CUDA available"
        if mps_available():
            return "mps", "auto: no CUDA, Apple MPS available"
        return "cpu", "auto: no CUDA or MPS available"
    ok = (spec == "cpu" or (spec.startswith("cuda") and cuda_available()) or (spec == "mps" and mps_available()))
    if spec.startswith("cuda:") and ok:
        idx = int(spec.split(":", 1)[1])
        ok = idx < _torch().cuda.device_count()
    if ok:
        return spec, f"requested {spec}"
    if allow_fallback:
        dev, why = resolve_device("auto")
        return dev, f"requested {spec} unavailable -> fallback ({why})"
    raise DeviceUnavailableError(
        f"Requested device '{spec}' is not available on this machine "
        f"(available accelerators: {accelerators() or 'none'}). Use --device auto/cpu, "
        f"or --allow-fallback for smoke tests.")


def is_cuda(device: str) -> bool:
    return device.startswith("cuda")


# ----------------------------------------------------------------------------- sync / memory
def synchronize(device: str) -> None:
    """Block until queued work on ``device`` has finished (no-op on CPU)."""
    if is_cuda(device):
        _torch().cuda.synchronize()
    elif device == "mps":
        _torch().mps.synchronize()


def reset_peak_memory(device: str) -> None:
    if is_cuda(device):
        _torch().cuda.reset_peak_memory_stats()


class AcceleratorMemoryTracker:
    """Peak accelerator memory. CUDA: ``max_memory_allocated``; MPS: the peak of
    ``driver_allocated_memory()`` sampled via :meth:`sample`; CPU: ``None``."""

    def __init__(self, device: str) -> None:
        self.device, self._mps_peak = device, 0
        reset_peak_memory(device)

    def sample(self) -> None:
        if self.device == "mps":
            self._mps_peak = max(self._mps_peak, _torch().mps.driver_allocated_memory())

    def peak_mb(self) -> float | None:
        return peak_accelerator_memory_mb(self.device, self._mps_peak)


def peak_accelerator_memory_mb(device: str, mps_sampled_peak_bytes: int = 0) -> float | None:
    if is_cuda(device):
        return _torch().cuda.max_memory_allocated() / 2**20
    if device == "mps":
        return max(mps_sampled_peak_bytes, _torch().mps.driver_allocated_memory()) / 2**20
    return None


# ----------------------------------------------------------------------------- seeding / determinism / threads
def seed_everything(seed: int, device: str = "cpu") -> None:
    """Seed python, numpy and (if loaded or needed) torch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    if "torch" in sys.modules or device != "cpu":
        torch = _torch()
        torch.manual_seed(seed)
        if is_cuda(device):
            torch.cuda.manual_seed_all(seed)


def set_determinism(device: str) -> dict[str, Any]:
    """Best-effort determinism; returns notes recorded in every result file."""
    notes: dict[str, Any] = {}
    if is_cuda(device):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch = _torch()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        notes["determinism"] = "cudnn.deterministic + use_deterministic_algorithms(warn_only)"
        notes["determinism_caveat"] = "cuDNN RNN kernels may remain slightly nondeterministic"
    elif device == "mps":
        notes["determinism"] = "best-effort (MPS has no deterministic mode)"
    else:
        notes["determinism"] = "cpu"
    return notes


def set_threads(n: int | None = None) -> int:
    """Apply the thread limit to BLAS/OpenMP env vars and to torch if it is loaded."""
    n = int(n or configured_threads())
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(n)
    return n


# ----------------------------------------------------------------------------- data movement
def to_device(x, device: str):
    """``x.to(device)`` for a tensor or module (tensors non-blocking on CUDA).

    ``device`` is the string resolved by :func:`resolve_device` (``cuda``, ``cuda:N``,
    ``mps`` or ``cpu``), so the same code runs on any hardware.
    """
    if hasattr(x, "parameters"):  # nn.Module
        return x.to(device)
    return x.to(device, non_blocking=is_cuda(device))


def module_device(model) -> str:
    """Device the model's parameters actually live on (e.g. ``cuda:0``, ``mps:0``, ``cpu``)."""
    return str(next(model.parameters()).device)


def assert_on_device(model, device: str) -> str:
    """Verify that ``model.to(device)`` took effect; return the actual device string."""
    actual = module_device(model)
    if actual.split(":")[0] != device.split(":")[0]:
        raise RuntimeError(f"model is on {actual}, expected {device}")
    return actual


def loader_kwargs(device: str) -> dict[str, Any]:
    """DataLoader settings: no worker processes; pinned memory only on CUDA."""
    return {"num_workers": 0, "pin_memory": is_cuda(device)}


@contextlib.contextmanager
def autocast(device: str, enabled: bool) -> Iterator[None]:
    """fp16 autocast on CUDA when ``enabled`` (experiments only; reported runs are fp32)."""
    if enabled:
        if not is_cuda(device):
            raise ValueError("--amp is only supported on CUDA")
        with _torch().autocast("cuda", dtype=_torch().float16):
            yield
    else:
        yield


@contextlib.contextmanager
def record_mps_fallbacks(device: str, sink: list[str]) -> Iterator[None]:
    """Collect PyTorch's "falls back to CPU" warnings emitted on MPS into ``sink``."""
    if device != "mps":
        yield
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    for w in caught:
        msg = str(w.message)
        if "fall back" in msg.lower() or "fallback" in msg.lower():
            if msg not in sink:
                sink.append(msg)
        else:  # re-emit unrelated warnings
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)


# ----------------------------------------------------------------------------- environment info
def _cmd(args: list[str]) -> str | None:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=60)
        return out.stdout.strip() or None
    except Exception:
        return None


def _cpu_model() -> str:
    if sys.platform == "darwin":
        return _cmd(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def hardware_info() -> dict[str, Any]:
    """Full hardware/software description (saved to ``results/env/hardware.json``)."""
    import psutil

    torch = _torch()
    info: dict[str, Any] = {
        "os": platform.platform(), "python": sys.version.split()[0], "cpu_model": _cpu_model(),
        "cpu_cores_physical": psutil.cpu_count(logical=False), "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1), "torch": torch.__version__,
        "cuda_available": cuda_available(), "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available() else None,
        "mps_available": mps_available(), "gpu_name": None, "gpu_memory_gb": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "n_threads": configured_threads(),
    }
    if cuda_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu_name"], info["gpu_memory_gb"] = p.name, round(p.total_memory / 2**30, 1)
        if shutil.which("nvidia-smi"):
            info["nvidia_smi"] = _cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                       "--format=csv,noheader"])
    elif mps_available():
        info["gpu_name"] = f"{info['cpu_model']} GPU (MPS)"
    for mod in ("sklearn", "xgboost", "kiwipiepy", "numpy", "pandas", "gradio"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    info["mamba_ssm_available"] = mamba_ssm_available()
    info["fingerprint"] = fingerprint(info)
    return info


def fingerprint(info: dict[str, Any]) -> str:
    """Short hash identifying the machine (CPU, GPU, OS family, core count)."""
    key = "|".join(str(info.get(k)) for k in ("cpu_model", "gpu_name", "cpu_cores_logical", "ram_gb"))
    key += "|" + platform.system()
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def compact_info() -> dict[str, Any]:
    """Compact hardware block stored in every run JSON."""
    full = hardware_info()
    keys = ("fingerprint", "os", "cpu_model", "gpu_name", "cuda_version", "cudnn_version", "torch", "n_threads")
    return {k: full.get(k) for k in keys}


def mamba_ssm_available() -> bool:
    """True if the official CUDA Mamba kernels are importable and CUDA is present."""
    try:
        import mamba_ssm  # noqa: F401
    except Exception:
        return False
    return cuda_available()


def resolve_mamba_backend(spec: str | None, device: str) -> str:
    """``auto`` -> ``mamba_ssm`` on CUDA when importable, else ``torch_ref``."""
    spec = spec or str(global_cfg().get("mamba_backend", "auto"))
    if spec == "auto":
        return "mamba_ssm" if is_cuda(device) and mamba_ssm_available() else "torch_ref"
    if spec == "mamba_ssm" and not (is_cuda(device) and mamba_ssm_available()):
        raise DeviceUnavailableError("mamba_backend=mamba_ssm requires CUDA and the mamba-ssm package")
    if spec not in ("mamba_ssm", "torch_ref"):
        raise ValueError(f"unknown mamba backend {spec!r}")
    return spec


def machine_details() -> dict[str, Any]:
    """Extra machine facts for ``results/env/hardware.json`` (not part of the fingerprint)."""
    import re

    d: dict[str, Any] = {}
    try:
        rel = dict(line.split("=", 1) for line in open("/etc/os-release").read().splitlines() if "=" in line)
        d["os_pretty"] = rel.get("PRETTY_NAME", "").strip('"')
    except OSError:
        d["os_pretty"] = platform.platform()
    d["disk_free_gb"] = {}
    for mnt in ("/", "/dev/shm", str(Path(__file__).resolve().parents[2])):
        if os.path.exists(mnt):
            u = shutil.disk_usage(mnt)
            d["disk_free_gb"][mnt] = {"total": round(u.total / 2**30, 1), "free": round(u.free / 2**30, 1)}
    if shutil.which("nvidia-smi"):
        rows = (_cmd(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"])
                or "").strip().splitlines()
        d["gpus"] = [dict(zip(("index", "name", "memory", "driver_version"), (x.strip() for x in r.split(","))))
                     for r in rows]
        d["gpu_count"] = len(rows)
        d["driver_version"] = d["gpus"][0]["driver_version"] if rows else None
        m = re.search(r"CUDA Version:\s*([0-9.]+)", _cmd(["nvidia-smi"]) or "")
        d["driver_max_cuda"] = m.group(1) if m else None
    else:
        d.update(gpus=[], gpu_count=0, driver_version=None, driver_max_cuda=None)
    nvcc = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else None)
    d["nvcc"] = {"path": nvcc, "version": (_cmd([nvcc, "--version"]) or "").strip().splitlines()[-1] if nvcc else None}
    return d


def dump_env(out_dir: str) -> dict[str, Any]:
    """Write hardware.json, conda_env.yaml and pip_freeze.txt."""
    import json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = {**hardware_info(), **machine_details()}
    torch = _torch()
    (out / "torch_versions.json").write_text(json.dumps({
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda_available() else None,
        "cuda_available": cuda_available(), "device_count": torch.cuda.device_count() if cuda_available() else 0,
        "mamba_backend_config": global_cfg().get("mamba_backend"), "mamba_ssm_available": mamba_ssm_available(),
    }, indent=2))
    (out / "hardware.json").write_text(json.dumps(info, indent=2))
    (out / "conda_env.yaml").write_text((_cmd(["conda", "env", "export", "-n", "ML"]) or "") + "\n")
    (out / "pip_freeze.txt").write_text((_cmd([sys.executable, "-m", "pip", "freeze"]) or "") + "\n")
    return info


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Print detected hardware; optionally dump env info.")
    ap.add_argument("--dump-env", metavar="DIR")
    ap.add_argument("--accelerators", action="store_true", help="print available accelerators, one per line")
    args = ap.parse_args()
    if args.accelerators:
        print("\n".join(accelerators()))
        return
    info = dump_env(args.dump_env) if args.dump_env else hardware_info()
    dev, why = resolve_device("auto")
    print(json.dumps(info, indent=2))
    print(f"resolved device: {dev} ({why}); threads: {configured_threads()}")


if __name__ == "__main__":
    main()
