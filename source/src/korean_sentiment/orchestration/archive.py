from __future__ import annotations

import tarfile
from datetime import datetime, timezone
from pathlib import Path

from korean_sentiment.config import CONFIG_DIR, PROJECT_ROOT, data_dir, logs_dir, results_dir


def archive_results() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = results_dir() / "archives" / f"korean_sentiment_results_{timestamp}.tar.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    roots = [CONFIG_DIR, data_dir() / "manifests", data_dir() / "splits", results_dir() / "csv", logs_dir(),
             results_dir() / "plans", results_dir() / "reports", results_dir() / "progress", results_dir() / "server",
             PROJECT_ROOT / "scripts", PROJECT_ROOT / "source" / "src", PROJECT_ROOT / "source" / "tests"]
    with tarfile.open(destination, "w:gz") as archive:
        for root in roots:
            if root.exists():
                for path in sorted(root.rglob("*")):
                    if path.is_file() and not any(part == "__pycache__" or part.endswith(".egg-info") for part in path.parts) and path.suffix != ".pyc" and path.name not in {"conda_history.yaml", "conda_explicit.txt", "pip_freeze.txt", "archive_manifest.json"}:
                        archive.add(path, arcname=str(path.relative_to(PROJECT_ROOT)), recursive=False)
        for name in ("README.md", "run_all.sh", "pyproject.toml", "requirements-base.txt", "requirements-cloud.txt"):
            path = PROJECT_ROOT / name
            if path.is_file():
                archive.add(path, arcname=name, recursive=False)
        for filename in ("record.json", "status.json", "run_metadata.json", "resolved_config.json", "resources.json", "metrics.json", "error.json", "confusion_matrix.json", "SUCCESS", "run_*.log"):
            for path in sorted((results_dir() / "run_records").glob(f"*/{filename}")):
                archive.add(path, arcname=str(path.relative_to(PROJECT_ROOT)))
    return destination
