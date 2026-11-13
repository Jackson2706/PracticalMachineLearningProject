"""Read only this project's CUDA compiler processes; omit unrelated commands."""
import json
from datetime import datetime, timezone
from pathlib import Path

import psutil

output = []
for process in psutil.process_iter(["pid", "name", "cmdline", "cpu_times", "memory_info", "status", "create_time"]):
    try:
        info = process.info
        tokens = info["cmdline"] or []
        name = info["name"]
        if name not in {"nvcc", "cicc", "ptxas", "cc1plus", "ninja", "g++", "c++"}:
            continue
        if not any("causal_conv1d" in token for token in tokens) and "causal_conv1d" not in process.cwd():
            continue
        output.append({"pid": info["pid"], "name": name, "status": info["status"],
                       "cpu_seconds": info["cpu_times"].user + info["cpu_times"].system,
                       "rss_bytes": info["memory_info"].rss,
                       "started_at": datetime.fromtimestamp(info["create_time"], timezone.utc).isoformat(),
                       "source_files": [Path(t).name for t in tokens if "causal_conv1d" in t]})
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue
path = Path(__file__).resolve().parents[1] / "source/results/server/compiler_processes.json"
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "compiler_processes": output,
                                 "ram_available": psutil.virtual_memory().available}, indent=2))
temporary.replace(path)
