from __future__ import annotations

import tarfile
from datetime import datetime, timezone
from pathlib import Path

from korean_sentiment.config import CONFIG_DIR, PROJECT_ROOT, data_dir, logs_dir, results_dir


def archive_results() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = results_dir() / "archives" / f"korean_sentiment_results_{timestamp}.tar.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    roots = [CONFIG_DIR, data_dir() / "manifests", data_dir() / "splits", results_dir() / "csv", logs_dir()]
    with tarfile.open(destination, "w:gz") as archive:
        for root in roots:
            if root.exists():
                archive.add(root, arcname=str(root.relative_to(PROJECT_ROOT)), recursive=True)
        for filename in ("record.json", "status.json", "run_metadata.json", "resolved_config.json", "resources.json", "metrics.json", "error.json"):
            for path in sorted((results_dir() / "run_records").glob(f"*/{filename}")):
                archive.add(path, arcname=str(path.relative_to(PROJECT_ROOT)))
    return destination
