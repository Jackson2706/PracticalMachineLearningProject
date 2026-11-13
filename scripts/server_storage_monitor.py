"""Sample allocated study storage without reading model/data file contents."""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "source/results/server"
sys.path.insert(0, str(ROOT / "source/src"))


def allocated_bytes(roots):
    seen = set()
    total = 0
    for root in roots:
        if not root.exists():
            continue
        for directory, children, files in os.walk(root, followlinks=False):
            children[:] = [name for name in children if name != "archives"]
            for name in files:
                try:
                    stat = (Path(directory) / name).lstat()
                except FileNotFoundError:
                    continue
                identity = (stat.st_dev, stat.st_ino)
                if identity not in seen:
                    seen.add(identity)
                    total += stat.st_blocks * 512
    return total


def sample():
    from korean_sentiment.utils.io import atomic_write_csv, atomic_write_json
    started = time.perf_counter()
    roots = [ROOT / "source/data", ROOT / "source/results", ROOT / "source/checkpoints"]
    allocated = allocated_bytes(roots)
    free = shutil.disk_usage(ROOT).free
    peak_path = SERVER / "storage_peak.json"
    old = json.loads(peak_path.read_text()) if peak_path.exists() else {}
    now = datetime.now(timezone.utc).isoformat()
    value = {"utc": now, "allocated_study_bytes": allocated, "disk_free_bytes": free,
             "peak_sampled_study_bytes": max(allocated, old.get("peak_sampled_study_bytes", 0)),
             "minimum_sampled_disk_free_bytes": min(free, old.get("minimum_sampled_disk_free_bytes", free)),
             "interval_seconds": 300, "scan_wall_seconds": time.perf_counter()-started,
             "first_sample_at": old.get("first_sample_at", now),
             "scope": "Allocated file blocks under source/data, source/results and source/checkpoints; archives and installed environment excluded; hard links deduplicated. Discrete samples can miss brief temporary peaks. Filesystem free space includes unrelated filesystem use."}
    atomic_write_json(peak_path, value)
    history_path = SERVER / "storage_history.csv"
    history = []
    if history_path.exists():
        with history_path.open() as handle:
            history = list(csv.DictReader(handle))
    history.append({key: value[key] for key in ["utc", "allocated_study_bytes", "disk_free_bytes", "scan_wall_seconds"]})
    atomic_write_csv(history_path, history, list(history[-1]))
    print(json.dumps(value), flush=True)
    return value


def worker_alive(active):
    import psutil
    pid = active.get("current_worker_pid")
    if not pid:
        return False
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE and any("server_execution.py" in a for a in process.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def finalize():
    from korean_sentiment.orchestration.aggregate import aggregate_all
    from korean_sentiment.orchestration.archive import archive_results
    from korean_sentiment.utils.io import atomic_write_json, atomic_write_text, sha256_file
    aggregate_all()
    spec = importlib.util.spec_from_file_location("server_execution_final", ROOT / "scripts/server_execution.py")
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    complete = server.audit_report()
    storage = sample()
    timings = server.rows(SERVER / "stage_timings.csv")
    production = [r for r in timings if r["stage"] in {"tune", "full", "bonus-tune", "bonus-full"}]
    cpu_seconds = sum(float(r["child_cpu_sec"]) for r in production) if production and all(r.get("child_cpu_sec") not in (None, "") for r in production) else None
    records = [server.read(p) for p in (ROOT / "source/results/run_records").glob("*/record.json")]
    gpu_records = [r for r in records if r.get("phase") != "smoke" and r.get("status") == "completed" and str(r.get("device", "")).startswith("cuda")]
    missing_gpu_durations = [r["run_id"] for r in gpu_records if r.get("duration_sec") is None]
    gpu_wall = sum(float(r["duration_sec"]) for r in gpu_records) if not missing_gpu_durations else None
    cpu_hours = cpu_seconds/3600 if cpu_seconds is not None else None
    gpu_hours = gpu_wall/3600 if gpu_wall is not None else None
    operational = {"complete": complete, "recorded_production_child_cpu_hours": cpu_hours,
                   "completed_gpu_assigned_job_wall_hours": gpu_hours, "missing_gpu_duration_run_ids": missing_gpu_durations,
                   "gpu_hours_scope": "Sum of completed CUDA-assigned job elapsed durations, including loading/evaluation/serialization; not GPU utilization-hours.",
                   "storage": storage}
    atomic_write_json(SERVER / "operational_totals.json", operational)
    report = ROOT / "source/results/reports/SERVER_EXECUTION_REPORT.md"
    text = report.read_text()
    text = text.replace("Actual CPU process-hours and GPU utilization-hours are unavailable for production records; elapsed assigned-device hours must not be presented as those measurements. Peak storage was not continuously sampled and remains unavailable.",
                        "Measured child-process CPU time for recorded production phases and completed GPU-assigned job wall time are reported below. GPU utilization-hours and the instantaneous storage peak remain unavailable; discrete storage samples provide the observed peak.")
    cpu_display = f"{cpu_hours:.6f}" if cpu_hours is not None else "unavailable"
    gpu_display = f"{gpu_hours:.6f}" if gpu_hours is not None else "unavailable"
    text += (f"\n\nRecorded production child-process CPU hours: {cpu_display}. Completed GPU-assigned job wall hours: {gpu_display}; this is elapsed job time, not GPU utilization-hours.\n\n"
             f"Largest sampled allocated study storage: {storage['peak_sampled_study_bytes']/2**30:.6f} GiB, excluding archives and the installed environment. Smallest sampled filesystem free space: {storage['minimum_sampled_disk_free_bytes']/2**30:.6f} GiB. Sampling began {storage['first_sample_at']} at five-minute intervals, so earlier and brief transient peaks may be missed. Archive byte size is recorded separately in archive_manifest.json.\n\n"
             "Authoritative operational data: source/results/server/operational_totals.json, storage_peak.json, storage_history.csv and stage_timings.csv. The post-run report/audit were regenerated after the coordinator stopped; prior result archives are preserved.\n")
    atomic_write_text(report, text)
    server.snapshot(final=True)
    destination = archive_results()
    atomic_write_json(SERVER / "archive_manifest.json", {"path": str(destination.relative_to(ROOT)), "bytes": destination.stat().st_size,
                                                       "sha256": sha256_file(destination), "complete": complete,
                                                       "kind": "final operationally audited results" if complete else "incomplete results and diagnostic evidence"})
    atomic_write_json(SERVER / "postprocessing_status.json", {"finished_at": datetime.now(timezone.utc).isoformat(), "complete": complete, "archive": str(destination.relative_to(ROOT))})


def main():
    if os.environ.get("CONDA_DEFAULT_ENV") != "ML":
        raise RuntimeError("run inside Conda ML")
    import fcntl
    lock = (SERVER / "storage_monitor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        sample()
        state = json.loads((SERVER / "workflow_state.json").read_text())
        active = json.loads((SERVER / "active_run.json").read_text())
        if state.get("phase") in {"completed", "blocked"} and not worker_alive(active):
            finalize()
            return
        time.sleep(300)


if __name__ == "__main__":
    main()
