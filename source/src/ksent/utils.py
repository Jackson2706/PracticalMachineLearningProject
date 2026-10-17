"""Small shared helpers: JSON I/O and provenance (device logic lives in ksent.hardware)."""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from ksent.config import SOURCE_DIR


def git_commit() -> str | None:
    """Current git commit hash of the project (``None`` outside a repo)."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=SOURCE_DIR, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=SOURCE_DIR, capture_output=True, text=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return None


def now_iso() -> str:
    """Local timestamp in ISO-8601 format."""
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(obj: Any, path: str | Path) -> None:
    """Atomically write JSON (write to tmp then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
