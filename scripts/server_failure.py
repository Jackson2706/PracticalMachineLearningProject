"""Preserve a truthful failure handoff even when project dependencies are missing."""
import csv
import hashlib
import json
import os
import sys
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

root = Path(__file__).resolve().parents[1]
results = root / "source/results"
server = results / "server"
now = datetime.now(timezone.utc)


def atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


active_path = server / "active_run.json"
active = json.loads(active_path.read_text()) if active_path.exists() else {}
active.update(phase="blocked", stopped_at=now.isoformat(), exit_code=int(sys.argv[1]))
atomic(active_path, json.dumps(active, indent=2))
counts = defaultdict(Counter)
for name in ["experiment_plan.csv", "bonus_experiment_plan.csv"]:
    path = results / "csv" / name
    if path.exists():
        with path.open() as handle:
            for row in csv.DictReader(handle):
                status_path = results / "run_records" / row["run_id"] / "status.json"
                status = json.loads(status_path.read_text()).get("status", "unknown") if status_path.exists() else "pending"
                counts[row["phase"]][status] += 1
message = "\n\nLauncher stopped at " + now.isoformat() + " (Asia/Seoul " + now.astimezone(timezone(timedelta(hours=9))).isoformat() + "). Exit code: " + sys.argv[1] + ". See source/results/server/workflow.log and the newest stage log.\n\nVerified plan status counts: " + json.dumps({k: dict(v) for k,v in counts.items()}) + (". Actual plans have not been generated.\n" if not counts else ".\n")
message += "\nNo completion or scientific performance is claimed. Host GPU access was verified; the earlier sandbox NVIDIA failure does not establish a host-driver fault.\n\nAfter diagnosing and fixing the recorded cause, resume setup/validation with `bash scripts/server_setup_ml.sh` in tmux, or resume production only after all gates pass with `conda run --no-capture-output -n ML python scripts/server_execution.py run`.\n"
report = results / "reports/SERVER_EXECUTION_REPORT.md"
atomic(report, (report.read_text() if report.exists() else "# Server execution report\n") + message)
progress = results / "progress"
atomic(progress / "latest.md", "# Workflow stopped\n" + message)
atomic(progress / f"update_{now.strftime('%Y%m%dT%H%M%SZ')}.md", "# Workflow stopped\n" + message)
archive = results / "archives" / f"server_blocked_diagnostics_{now.strftime('%Y%m%dT%H%M%SZ')}.tar.gz"
archive.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(archive, "w:gz") as package:
    for folder in [server, results / "reports", progress, results / "plans", results / "csv"]:
        if folder.exists():
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.name not in {"conda_history.yaml", "conda_explicit.txt", "pip_freeze.txt", "workflow.lock"}:
                    package.add(path, arcname=str(path.relative_to(root)), recursive=False)
sha = hashlib.sha256()
with archive.open("rb") as handle:
    while data := handle.read(1024**2):
        sha.update(data)
atomic(server / "archive_manifest.json", json.dumps({"path": str(archive.relative_to(root)), "bytes": archive.stat().st_size, "sha256": sha.hexdigest(), "kind": "blocked diagnostic evidence"}, indent=2))
print(message, flush=True)
