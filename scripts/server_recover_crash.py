"""Run the interrupted condition once in a fresh process, preserving crash evidence."""
import csv
import faulthandler
import json
import os
from datetime import datetime, timezone
from pathlib import Path

faulthandler.enable(all_threads=True)
assert os.environ.get("CONDA_DEFAULT_ENV") == "ML"
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "4"
from korean_sentiment.orchestration.executor import execute_job
from korean_sentiment.utils.io import atomic_write_json

root = Path(__file__).resolve().parents[1]
server = root / "source/results/server"
run_id = "part1-a02ea6335bf777319573"
directory = root / "source/results/run_records" / run_id
previous = {p.name: p.read_text() for p in directory.iterdir() if p.is_file() and p.suffix == ".json"}
atomic_write_json(server / "crash_recovery_evidence.json", {
    "utc": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
    "original_exit": 139, "previous_metadata": previous,
    "diagnosis": "Native segmentation fault; no Python traceback. Cause not established. One isolated fresh-process reproduction with faulthandler, unchanged scientific configuration.",
})
with (root / "source/results/csv/experiment_plan.csv").open() as handle:
    row = next(r for r in csv.DictReader(handle) if r["run_id"] == run_id)
result = execute_job(row)
atomic_write_json(server / "crash_recovery_result.json", result)
print(json.dumps(result), flush=True)
if result["status"] not in {"completed", "skipped"}:
    raise SystemExit(1)
