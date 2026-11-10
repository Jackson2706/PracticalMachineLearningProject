from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from korean_sentiment.config import results_dir
from korean_sentiment.orchestration.aggregate import _read, collect_records
from korean_sentiment.orchestration.state import run_directory
from korean_sentiment.utils.io import read_json


def status_report() -> dict[str, Any]:
    records = {row["run_id"]: row for row in collect_records()}
    plans = _read(results_dir() / "csv" / "experiment_plan.csv") + _read(results_dir() / "csv" / "bonus_experiment_plan.csv")
    overall = Counter()
    grouped: dict[str, Counter] = defaultdict(Counter)
    for row in plans:
        status_path = run_directory(row["run_id"]) / "status.json"
        status = read_json(status_path).get("status") if status_path.exists() else records.get(row["run_id"], {}).get("status", "pending")
        overall[status] += 1
        grouping = row.get("target_domain") or row.get("dataset") or "unknown"
        method = row["objective"] if row["phase"].startswith("bonus") else row["model"]
        grouped[f"{row['phase']}:{grouping}:{method}"][status] += 1
    overall["missing"] = overall.get("pending", 0)
    return {"overall": dict(sorted(overall.items())), "groups": {key: dict(value) for key, value in sorted(grouped.items())}}
