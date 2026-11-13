#!/usr/bin/env python3
"""Serial server gates, measured pilots and resumable execution in Conda ML."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
sys.path.insert(0, str(SOURCE / "src"))
SERVER = SOURCE / "results/server"
SEOUL = timezone(timedelta(hours=9))


def stamp():
    return datetime.now(timezone.utc)


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def rows(path):
    if not Path(path).exists():
        return []
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def write(path, value):
    from korean_sentiment.utils.io import atomic_write_json, atomic_write_text
    if isinstance(value, str):
        atomic_write_text(path, value)
    else:
        atomic_write_json(path, value)


def table(path, values):
    from korean_sentiment.utils.io import atomic_write_csv
    fields = list(dict.fromkeys(key for row in values for key in row))
    atomic_write_csv(path, values, fields or ["status"])


def summarize_costs(records, group_fields):
    """Derive matched-condition costs; missing resource fields stay missing."""
    import statistics
    grouped = defaultdict(list)
    for record in records:
        key = tuple(int(record[field]) if field in {"train_size", "target_train_size"} else record[field] for field in group_fields)
        grouped[key].append(record)
    output = []
    mean_fields = ["fit_time_sec", "classifier_latency_ms_p50", "end_to_end_latency_ms_p50", "serialized_size_mb",
                   "deployed_parameter_count", "same_sentiment_cross_domain_cosine", "different_sentiment_cross_domain_cosine",
                   "cross_domain_sentiment_separation_gap", "same_domain_same_sentiment_cosine"]
    for key, group in sorted(grouped.items()):
        row = dict(zip(group_fields, key))
        row.update(completed_seeds=len(group), device=",".join(sorted({str(r.get("device", "unavailable")) for r in group})))
        for field in mean_fields + ["peak_host_ram_mb", "peak_gpu_memory_mb"]:
            values = [float(r[field]) for r in group if r.get(field) not in (None, "")]
            valid = len(values) == len(group) and all(math.isfinite(v) for v in values)
            suffix = "_max" if field.startswith("peak_") else "_mean"
            row[field+suffix] = (max(values) if suffix == "_max" else statistics.fmean(values)) if valid else None
        output.append(row)
    return output


def phase(name):
    previous = read(SERVER / "workflow_state.json", {})
    write(SERVER / "workflow_state.json", {**previous, "phase": name, "updated_at": stamp().isoformat()})
    active = read(SERVER / "active_run.json", {})
    if active:
        if name != "blocked":
            active.pop("exit_code", None)
            active.pop("stopped_at", None)
        import psutil
        known = {"scripts/server_launcher.sh", "scripts/server_setup_ml.sh", "scripts/server_rebuild_mamba.sh", "scripts/server_resume_verified.sh", "scripts/server_resume_after_crash.sh"}
        for parent in psutil.Process().parents():
            try:
                script = next((token for token in parent.cmdline() if token in known), None)
                if script:
                    active.update(launcher_pid=parent.pid, command=f"bash {script}",
                                  start_time=datetime.fromtimestamp(parent.create_time(), timezone.utc).isoformat())
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        write(SERVER / "active_run.json", {**active, "phase": name, "updated_at": stamp().isoformat(),
                                         "current_worker_pid": os.getpid(),
                                         "production_started": active.get("production_started", False) or name in {"tune", "full", "bonus-tune", "bonus-full"}})
    print(f"{stamp().isoformat()} phase={name}", flush=True)


def command(args, label=None):
    import resource
    name = label or args[1]
    phase(name)
    started_at = stamp()
    clock = time.perf_counter()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    path = SERVER / f"{name}_{stamp().strftime('%Y%m%dT%H%M%SZ')}.log"
    with path.open("w") as log:
        completed = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    timings = rows(SERVER / "stage_timings.csv")
    timings.append({"stage": name, "started_at": started_at.isoformat(), "finished_at": stamp().isoformat(),
                    "wall_sec": time.perf_counter()-clock, "child_cpu_sec": usage_after.ru_utime+usage_after.ru_stime-usage_before.ru_utime-usage_before.ru_stime,
                    "exit_code": completed.returncode, "log": str(path.relative_to(ROOT))})
    table(SERVER / "stage_timings.csv", timings)
    if completed.returncode:
        raise RuntimeError(f"{args!r} failed with exit {completed.returncode}; see {path}")
    return path


def plans():
    return rows(SOURCE / "results/csv/experiment_plan.csv") + rows(SOURCE / "results/csv/bonus_experiment_plan.csv")


def verify_plans(preview=False):
    from korean_sentiment.config import FINAL_SEEDS
    from korean_sentiment.orchestration.planner import build_bonus_plans
    if preview:
        build_bonus_plans()  # Read-only dependency preview; no bonus training is permitted here.
    plan = plans()
    assert len({r["run_id"] for r in plan}) == len(plan), "duplicate run IDs"
    counts = Counter(r["phase"] for r in plan)
    assert counts["part1_tuning"] == 270
    assert counts["bonus_tuning"] == 36 and counts["bonus_final"] == 108
    for part in ("part1_final", "bonus_final"):
        assert {int(r["seed"]) for r in plan if r["phase"] == part} == set(FINAL_SEEDS)
        conditions = defaultdict(set)
        for r in plan:
            if r["phase"] == part:
                conditions[(r["dataset"], r["model"], r["objective"], r["train_size"])].add(int(r["seed"]))
        assert all(seeds == set(FINAL_SEEDS) for seeds in conditions.values()), "condition has missing final seeds"
    p1 = {r["run_id"]: r for r in plan if r["phase"] == "part1_final"}
    refs = rows(SOURCE / "results/plans/bonus_reuse_references.csv")
    assert len(refs) == 72
    for ref in refs:
        ids = ref["candidate_run_ids"].split(",")
        assert ids and all(rid in p1 for rid in ids)
        assert len(ids) == (4 if ref["reference_type"] == "best_tfidf" else 1)
        for rid in ids:
            value = p1[rid]
            assert (value["dataset"], value["train_size"], value["seed"]) == (ref["target_domain"], ref["target_train_size"], ref["seed"])
    summary = read(SOURCE / "results/plans/part1_plan_summary.json")
    assert counts["part1_final"] == sum(len(grid) for grid in summary["dataset_schedules"].values()) * 9 * 3
    text = "# Plan verification\n\n" + json.dumps({"counts": dict(counts), "total": len(plan), "part1": summary}, indent=2)
    text += "\n\nRun IDs unique; final seeds 13,42,2026; 72 reference rows link existing Part 1 conditions without new jobs. Training loaders use train-only manifests; tuning loaders omit tests."
    text += "\n\nBonus plans are dependency previews pending frozen Part 1 settings; IDs will be regenerated after selection.\n" if preview else "\n\nBonus dependencies have been checked before execution.\n"
    write(SERVER / "plan_verification.md", text)
    return counts


def exports():
    for filename, args in [
        ("conda_history.yaml", ["conda", "env", "export", "-n", "ML", "--from-history"]),
        ("conda_explicit.txt", ["conda", "list", "-n", "ML", "--explicit"]),
        ("pip_freeze.txt", [sys.executable, "-m", "pip", "freeze"]),
    ]:
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        write(SERVER / filename, result.stdout)
    import importlib.metadata
    import torch
    from korean_sentiment.orchestration.preflight import mamba_smoke
    check = {"utc": stamp().isoformat(), "executable": sys.executable, "python": sys.version,
             "cuda_available": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0),
             "cuda_tensor_sum": torch.ones(16, device="cuda").sum().item(), "mamba_forward_backward": mamba_smoke(),
             "packages": {name: importlib.metadata.version(name) for name in ["torch", "scikit-learn", "xgboost", "mamba-ssm", "causal-conv1d", "pytest"]}}
    write(SERVER / "dependency_check.md", "# Verified ML dependencies\n\n```json\n" + json.dumps(check, indent=2) + "\n```\n\nInitial failure logs are preserved. GPU checks require host access outside the restricted execution sandbox.\n")
    inventory = read(SERVER / "server_inventory.json", {})
    write(SERVER / "server_inventory.json", {**inventory, "validated_dependencies": check})
    sources = read(SERVER / "dependency_sources.json", {})
    write(SERVER / "dependency_sources.json", {**sources, "installed_and_verified": True,
          "mamba_installation": "2.2.6.post3 rebuilt locally using CUDA 12.1 for glibc 2.31",
          "causal_conv1d_installation": "1.6.0 rebuilt locally; required cpp_functions API",
          "build_logs": ["mamba_source_build.log", "causal_source_build.log"]})


def prepare():
    import psutil
    matched = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            tokens = process.info["cmdline"] or []
            markers = [token for token in tokens if token in {"korean_sentiment.cli", "ksent.sweep", "ksent.run_one", "ksent.tune"}]
            if markers and process.pid != os.getpid():
                matched.append({"pid": process.pid, "experiment_markers": markers})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    write(SERVER / "host_experiment_process_check.json", {"utc": stamp().isoformat(), "matching_processes": matched})
    assert not matched, "another experiment process is active; inspect the restricted marker inventory"
    command([str(ROOT / "run_all.sh"), "preflight", "--cloud-neural"])
    command([str(ROOT / "run_all.sh"), "test"])
    command([sys.executable, "-m", "pytest", "-q", "source/tests/test_server_safety.py"], "server_tests")
    for script in [ROOT / "run_all.sh", *list((ROOT / "scripts").glob("*.sh"))]:
        assert os.access(script, os.X_OK), f"not executable: {script}"
        assert "set -euo pipefail" in script.read_text(), f"missing strict shell handling: {script}"
    command([str(ROOT / "run_all.sh"), "download"])
    command([str(ROOT / "run_all.sh"), "preprocess"])
    command([str(ROOT / "run_all.sh"), "plan"])
    verify_plans(preview=True)
    stats = rows(SOURCE / "results/csv/dataset_statistics.csv")
    reports = {p.stem: read(p) for p in (SOURCE / "data/manifests").glob("*_cleaning_report.json")}
    from korean_sentiment.data.prepare import load_split
    balance = {}
    for stat in stats:
        balance[stat["dataset"]] = {}
        for split in ["train", "validation", "test"]:
            frame = load_split(stat["dataset"], split)
            balance[stat["dataset"]][split] = {str(label): int((frame["label"] == label).sum()) for label in [0, 1]}
    write(SERVER / "preprocessing_report.md", "# Preprocessing\n\n```json\n" + json.dumps({"statistics": stats, "cleaning": reports, "split_class_balance": balance}, indent=2) + "\n```\n\nChecksums: source/data/manifests/download_manifest.csv. Splits: source/data/splits/. Nested orders and data hash: source/data/manifests/. Bonus uses target prefixes 200/500/1000/2000 and source prefixes 2000 from each other domain.\n")
    exports()
    write(SERVER / "preparation_passed.json", {"utc": stamp().isoformat(), "tests": "documented suite plus server safety tests", "preflight": "cloud-neural", "plans": "verified dependency preview"})


def memory_probe(name):
    import numpy as np
    import torch
    from korean_sentiment.config import model_configs
    from korean_sentiment.models import make_pipeline, load_pipeline
    from korean_sentiment.orchestration.state import validate_completed
    from korean_sentiment.orchestration.executor import execute_job
    from korean_sentiment.evaluation.metrics import classification_metrics

    objective = name if name in {"pooled_ce", "supcon", "db_cdsupcon"} else "ce"
    model = "transformer" if objective != "ce" else name
    config = model_configs(model)[-1]
    # Full sequence length and near-cap vocabulary; production architecture is unchanged.
    n = 384 if objective != "ce" else 192
    texts = [" ".join(f"토큰{(i * 128 + j) % 20000}" for j in range(128)) for i in range(n)]
    labels = np.arange(n) % 2
    domains = np.arange(n) // (n // 3) if objective != "ce" else None
    pipeline = make_pipeline(model, config, 13, "cuda", 4)
    fitted = pipeline.fit(texts, labels, texts[:12], labels[:12], domains=domains,
                          objective=objective, effective_batch_size=48 if domains is not None else 64,
                          contrastive_lambda=0.1, contrastive_temperature=0.07, cross_domain_weight=4.0,
                          target_domain_id=0 if domains is not None else None, max_epochs=1, patience=1)
    predictions = pipeline.predict(texts[:12])
    metrics = classification_metrics(labels[:12], predictions)
    with tempfile.TemporaryDirectory(prefix="ks_probe_") as tmp:
        artifact = pipeline.save(Path(tmp) / name)
        loaded = load_pipeline(artifact, "cuda")
        assert np.array_equal(predictions, loaded.predict(texts[:12]))
    allocated = torch.cuda.max_memory_allocated() / 1024**2
    reserved = torch.cuda.max_memory_reserved() / 1024**2
    assert reserved < torch.cuda.get_device_properties(0).total_memory / 1024**2 * 0.85, "insufficient VRAM headroom"
    directory = SOURCE / "results/run_records" / f"smoke-{name}"
    assert validate_completed(directory), "invalid smoke run record"
    assert execute_job({"run_id": f"smoke-{name}", "data_manifest_hash": None, "config_hash": None})["status"] == "skipped"
    result = {"pipeline": name, "status": "passed", "fit": True, "prediction": True, "metrics": True,
              "serialization": True, "reload": True, "resumption": True, "run_record": True,
              "peak_allocated_mib": allocated, "peak_reserved_mib": reserved,
              "physical_batch_size": 48 if domains is not None else 64,
              "effective_batch_size": 48 if domains is not None else 64,
              "precision": os.environ["KS_PRECISION"], "finite_validation_macro_f1": metrics["macro_f1"],
              "optimizer_steps": fitted["optimizer_steps"]}
    write(SERVER / "probes" / f"{name}.json", result)


def smoke():
    command([str(ROOT / "run_all.sh"), "smoke", "--device", "cuda"])
    from korean_sentiment.config import NEURAL_MODELS
    from korean_sentiment.orchestration.state import validate_completed
    from korean_sentiment.orchestration.executor import execute_job
    matrix = []
    for item in rows(SOURCE / "results/csv/smoke_status.csv"):
        assert item["status"] == "completed", item
        name = item["pipeline"]
        if name in NEURAL_MODELS or name in {"pooled_ce", "supcon", "db_cdsupcon"}:
            command([sys.executable, str(Path(__file__)), "probe", "--name", name], f"memory_probe_{name}")
            matrix.append(read(SERVER / "probes" / f"{name}.json"))
        else:
            assert validate_completed(SOURCE / "results/run_records" / f"smoke-{name}")
            assert execute_job({"run_id": f"smoke-{name}", "data_manifest_hash": None, "config_hash": None})["status"] == "skipped"
            matrix.append({"pipeline": name, "status": "passed", "fit": True, "prediction": True,
                           "metrics": True, "serialization": True, "reload": True, "resumption": True,
                           "run_record": True, "peak_allocated_mib": None, "peak_reserved_mib": None,
                           "memory_measurement": "CPU pipeline; GPU not applicable"})
    assert len(matrix) == 12
    table(SERVER / "smoke_matrix.csv", matrix)
    write(SERVER / "smoke_report.md", "# Smoke and batch probes\n\nAll twelve pipelines passed fit, prediction, metric, serialization/reload and completed-record resumption checks. Neural probes used full length 128 and the largest configured architecture, physical/effective batches 64 (Part 1) or 48 (pooled objectives), without accumulation or scientific reductions. See smoke_matrix.csv for allocated/reserved VRAM and precision. These are synthetic stability checks, not scientific performance results.\n")


def pilot_one(spec):
    import numpy as np
    from korean_sentiment.config import model_configs
    from korean_sentiment.data.prepare import load_subset, load_split
    from korean_sentiment.models import make_pipeline, load_pipeline
    from korean_sentiment.evaluation.measure import measure_pipeline, PeakRSS, artifact_size_mb
    from korean_sentiment.evaluation.metrics import classification_metrics
    from korean_sentiment.utils.runtime import configure_threads
    threads = configure_threads()
    started = time.perf_counter()
    cpu_started = time.process_time()
    model, dataset, size, objective = spec["model"], spec["dataset"], spec["size"], spec["objective"]
    train = load_subset(dataset, 777, size)
    validation = load_split(dataset, "validation")
    domains = None
    if objective != "ce":
        import pandas as pd
        others = [d for d in ("nsmc", "shopping", "steam") if d != dataset]
        frames = [train] + [load_subset(d, 777, 2000) for d in others]
        domains = np.concatenate([np.full(len(frame), i) for i, frame in enumerate(frames)])
        train = pd.concat(frames, ignore_index=True)
    config = model_configs(model)[0]
    device = "cpu" if model.startswith("tfidf_") else "cuda"
    pipeline = make_pipeline(model, config, 777, device, threads)
    with PeakRSS() as rss:
        fitted = pipeline.fit(train.text.tolist(), train.label.to_numpy(), validation.text.tolist(), validation.label.to_numpy(),
                              domains=domains, objective=objective, contrastive_lambda=0.1, contrastive_temperature=0.07,
                              cross_domain_weight=4.0, target_domain_id=0 if domains is not None else None,
                              effective_batch_size=48 if domains is not None else None, max_epochs=2, patience=2)
        inference_started = time.perf_counter()
        metrics = classification_metrics(validation.label, pipeline.predict(validation.text.tolist()))
        validation_seconds = time.perf_counter() - inference_started
        latency = measure_pipeline(pipeline, validation.text.head(32).tolist(), 10, 50, 128)
        with tempfile.TemporaryDirectory(prefix="ks_pilot_") as tmp:
            artifact = pipeline.save(Path(tmp) / model)
            size_mb = artifact_size_mb(artifact)
            loaded = load_pipeline(artifact, device)
            assert np.array_equal(pipeline.predict(validation.text.head(8).tolist()), loaded.predict(validation.text.head(8).tolist()))
    value = {**spec, "duration_sec": time.perf_counter() - started, "training": fitted,
             "cpu_process_sec": time.process_time() - cpu_started,
             "validation_evaluation_sec": validation_seconds, "serialized_size_mb": size_mb,
             "latency": latency, "peak_host_ram_mb": rss.megabytes, "device": device,
             "validation_macro_f1": metrics["macro_f1"], "production_result": False,
             "note": "Non-scientific pilot: two neural epochs, config candidate 1, seed777; no test evaluation, no selected configuration."}
    write(SERVER / "pilots" / f"{spec['id']}.json", value)


def pilot():
    from korean_sentiment.config import MODELS
    command([str(ROOT / "run_all.sh"), "pilot", "--dataset", "nsmc", "--model", "transformer", "--max-jobs", "1", "--device", "cuda"], "repository_pilot")
    stats = rows(SOURCE / "results/csv/dataset_statistics.csv")
    specs = []
    for stat in stats:
        d, full = stat["dataset"], int(stat["n_full"])
        for model in MODELS:
            for size in [200, 8000] + ([full] if model in {"tfidf_rf", "mamba", "transformer"} else []):
                specs.append({"id": f"{d}_{model}_{size}", "dataset": d, "model": model, "size": size, "objective": "ce"})
        for objective in ["pooled_ce", "supcon", "db_cdsupcon"]:
            specs.append({"id": f"{d}_{objective}_500", "dataset": d, "model": "transformer", "size": 500, "objective": objective})
    write(SERVER / "pilot_plan.json", specs)
    for spec in specs:
        if not (SERVER / "pilots" / f"{spec['id']}.json").exists():
            command([sys.executable, str(Path(__file__)), "pilot-one", "--name", json.dumps(spec)], f"pilot_{spec['id']}")
    estimate()


def predict_seconds(job, measurements, percentile):
    from korean_sentiment.config import model_configs
    objective = job["objective"]
    dataset = job.get("target_domain") or job["dataset"]
    available = [p for p in measurements if p["dataset"] == dataset and p["model"] == job["model"] and p["objective"] == objective]
    if not available:
        raise RuntimeError(f"missing pilot stratum: {dataset}/{job['model']}/{objective}")
    n = int(job["train_size"])
    points = sorted(available, key=lambda p: p["size"])
    def curve(value, scale_single=False):
        # Fixed validation/startup costs must not be multiplied by n / n_pilot.
        samples = []
        for p in points:
            x = p["size"] + (4000 if objective != "ce" else 0)
            y = max(0.0, value(p))
            samples.append((x, max(y, samples[-1][1]) if samples else y))
        if len(samples) == 1:
            x, y = samples[0]
            return y * n / x if scale_single else y
        pair = next(((a, b) for a, b in zip(samples, samples[1:]) if n <= b[0]), (samples[-2], samples[-1]))
        a, b = pair
        return max(0.0, a[1] + (b[1]-a[1]) * (n-a[0]) / (b[0]-a[0]))
    overhead = curve(lambda p: p["validation_evaluation_sec"])
    if job["model_family"] == "neural":
        # The low/central/high cases include unknown early stopping and tuning widths.
        expected_epochs = {"low": 5, "central": 12, "high": 20}[percentile]
        configs = model_configs(job["model"])
        selected_id = job.get("config_id")
        if job["phase"].startswith("bonus"):
            frozen = rows(SOURCE / "results/csv/selected_configs.csv")
            selected_id = next((r["config_id"] for r in frozen if r["dataset"] == dataset and r["model"] == "transformer"), None)
        known_config = next((c for c in configs if c["id"] == selected_id), None)
        def operation_proxy(c):
            h, e, layers = c["hidden_dim"], c["embedding_dim"], c["num_layers"]
            if job["model"] == "avg_mlp":
                return 128*e + e*h + max(0, layers-1)*h*h
            if job["model"] in {"lstm", "gru"}:
                return h*(e+h) + max(0, layers-1)*3*h*h
            if job["model"] == "transformer":
                return layers*(4*128*h*h + 2*128*128*h + 2*128*h*h*c["ff_multiplier"])
            return layers*(3*h*h*c["expand"] + 2*h*c["expand"]*c["d_state"])
        ratios = [operation_proxy(c)/operation_proxy(configs[0]) for c in configs]
        width_multiplier = operation_proxy(known_config)/operation_proxy(configs[0]) if known_config else {
            "low": min(ratios), "central": sum(ratios)/len(ratios), "high": max(ratios)}[percentile]
        width_multiplier *= {"low": 0.65, "central": 1.0, "high": 1.5}[percentile]
        per_epoch = curve(lambda p: p["training"]["fit_time_sec"] / len(p["training"]["history"]), scale_single=True)
        outside_fit = curve(lambda p: p["duration_sec"] - p["training"]["fit_time_sec"])
        seconds = outside_fit + per_epoch * expected_epochs * width_multiplier
    else:
        seconds = curve(lambda p: p["duration_sec"])
        configs = model_configs(job["model"])
        config = next((c for c in configs if c["id"] == job["config_id"]), None)
        if job["model"] in {"tfidf_rf", "tfidf_xgb"}:
            trees = config["n_estimators"] if config else sum(c["n_estimators"] for c in configs) / len(configs)
            seconds *= trees / configs[0]["n_estimators"]
        seconds *= {"low": 0.75, "central": 1.25, "high": 2.5}[percentile]
    if job["phase"].endswith("final"):
        seconds += overhead * (3 if job["phase"].startswith("bonus") else 1)
    return seconds


def estimate():
    from korean_sentiment.orchestration.state import validate_completed
    measurements = [read(p) for p in (SERVER / "pilots").glob("*.json")]
    plan = plans()
    predicted = []
    for job in plan:
        predicted.append({"run_id": job["run_id"], "phase": job["phase"], "device": "cpu" if job["model_family"] == "tfidf" else "cuda",
                          "already_completed": validate_completed(SOURCE / "results/run_records" / job["run_id"], job["data_manifest_hash"], job["config_hash"]),
                          **{f"{case}_sec": predict_seconds(job, measurements, case) for case in ["low", "central", "high"]}})
    table(SERVER / "runtime_estimate.csv", predicted)
    totals = {case: sum(p[f"{case}_sec"] for p in predicted if not p["already_completed"]) * 1.05 for case in ["low", "central", "high"]}
    summary = []
    for part in sorted({p["phase"] for p in predicted}):
        for device in ["cpu", "cuda"]:
            subset = [p for p in predicted if p["phase"] == part and p["device"] == device]
            summary.append({"phase": part, "device": device, "jobs": len(subset), "already_completed": sum(p["already_completed"] for p in subset), **{f"{case}_hours": sum(p[f"{case}_sec"] for p in subset if not p["already_completed"]) / 3600 for case in totals}})
    result = {"generated_at": stamp().isoformat(), "pilot_jobs": measurements, "phase_hours": summary,
              "measured_stage_timings": rows(SERVER / "stage_timings.csv"),
              "repository_pilot_records": [read(SOURCE / "results/run_records" / job["run_id"] / "run_metadata.json") for job in plan if job["phase"] == "part1_tuning" and job["model"] == "transformer" and job["dataset"] == "nsmc" and (SOURCE / "results/run_records" / job["run_id"] / "SUCCESS").exists()],
              "wall_seconds": totals, "completion_asia_seoul": {case: (stamp() + timedelta(seconds=value)).astimezone(SEOUL).isoformat() for case, value in totals.items()},
              "central_exceeds_24h": totals["central"] > 86400, "safe_concurrency": 1,
              "assumptions": "Per-dataset/model/objective strata; monotone piecewise-linear size curves, separating measured neural per-epoch fit from loading/serialization/validation overhead. Neural 5/12/20-epoch scenarios; configuration-derived operation proxies for known architectures, or min/mean/max candidate proxies for unfrozen selections, with 0.65/1/1.5 GPU-efficiency uncertainty. Operation proxies are assumptions, not measured GPU FLOPs. Classical tree scaling and 0.75/1.25/2.5 uncertainty; final test overhead estimated from measured validation; 5% aggregation/monitor reserve. CPU/CUDA hours are assigned-device elapsed hours, not measured utilization-hours. Full-size pilots use RF, Transformer and Mamba candidate1. Other full-size families are extrapolated; hyperparameter depth/width and early stopping remain major uncertainty.",
              "comparison_5_9_days": "The initial 5–9 days is not an input to this calculation. Differences follow measured pilot strata, dataset sizes, early-stopping assumptions and conservative width/depth uncertainty."}
    write(SERVER / "runtime_estimate.json", result)
    text = "# Measured pilot runtime estimate\n\n" + json.dumps({k: v for k, v in result.items() if k != "pilot_jobs"}, indent=2) + "\n\nPilot details: runtime_estimate.json and server/pilots/. Per-run predictions: runtime_estimate.csv. Test overhead is an estimate based on validation; actual test evaluation remains restricted to final runs.\n"
    write(SERVER / "runtime_estimate.md", text)
    print(text, flush=True)


def snapshot(final=False):
    import psutil
    import statistics
    progress = SOURCE / "results/progress"
    progress.mkdir(parents=True, exist_ok=True)
    plan = plans()
    counters = Counter()
    grouped = defaultdict(Counter)
    current = []
    now = stamp()
    completed_recent = failed_recent = 0
    predictions = {r["run_id"]: r for r in rows(SERVER / "runtime_estimate.csv")}
    remaining = Counter({case: 0.0 for case in ["low", "central", "high"]})
    ratios = defaultdict(list)
    def stratum(job):
        return (job["phase"], job["dataset"], job["model"], job["objective"])
    for job in plan:
        prediction = predictions.get(job["run_id"], {})
        record = read(SOURCE / "results/run_records" / job["run_id"] / "record.json", {})
        if record.get("status") == "completed" and record.get("duration_sec") and prediction.get("central_sec"):
            ratios[stratum(job)].append(float(record["duration_sec"]) / float(prediction["central_sec"]))
    calibration = {key: statistics.median(values) for key, values in ratios.items()}
    errors = Counter()
    for job in plan:
        elapsed_job = 0
        directory = SOURCE / "results/run_records" / job["run_id"]
        status = read(directory / "status.json", {}).get("status", "pending")
        counters[status] += 1
        size = int(job["train_size"])
        group = f"{job['phase']}:{job.get('target_domain') or job['dataset']}:{job['objective'] if job['phase'].startswith('bonus') else job['model']}:{'small' if size<=2000 else 'medium' if size<=16000 else 'large'}"
        grouped[group][status] += 1
        meta = read(directory / "run_metadata.json", {})
        if status == "running":
            started = meta.get("started_at") or read(directory / "status.json", {}).get("started_at")
            elapsed_job = (now-datetime.fromisoformat(started)).total_seconds() if started else 0
            current.append({"run_id": job["run_id"], "elapsed_sec": elapsed_job if started else None})
        if meta.get("finished_at") and (now-datetime.fromisoformat(meta["finished_at"])).total_seconds() <= 43200:
            completed_recent += status == "completed"
            failed_recent += status in {"failed", "blocked"}
        if status in {"failed", "blocked"}:
            error = read(directory / "error.json", {})
            errors[f"{error.get('error_type')}: {error.get('message')}"] += 1
        if status != "completed":
            for case in ["low", "central", "high"]:
                remaining[case] += max(0, float(predictions.get(job["run_id"], {}).get(f"{case}_sec") or 0) * calibration.get(stratum(job), 1.0) - elapsed_job)
    state = read(SERVER / "workflow_state.json", {})
    result = {"utc": now.isoformat(), "asia_seoul": now.astimezone(SEOUL).isoformat(), "phase": state.get("phase"),
              "elapsed_sec": time.time()-state.get("started_epoch", time.time()), "counts": {k: counters[k] for k in ["completed", "running", "pending", "failed", "blocked", "stale", "interrupted"]},
              "total": len(plan), "percent": 100*counters["completed"]/len(plan) if plan else None,
              "counts_by_part_domain_model_size_group": {k: dict(v) for k, v in grouped.items()},
              "new_completed_last_12h": completed_recent, "new_failed_last_12h": failed_recent,
              "observed_jobs_per_hour_last_12h": completed_recent/min(12, max(1/3600, (time.time()-state.get('started_epoch', time.time()))/3600)),
              "current_jobs": current, "load_average": list(os.getloadavg()), "ram": psutil.virtual_memory()._asdict(),
              "disk_free_bytes": shutil.disk_usage(ROOT).free, "remaining_seconds": dict(remaining) if predictions else None,
              "estimated_completion_asia_seoul": {k: (now+timedelta(seconds=v)).astimezone(SEOUL).isoformat() for k,v in remaining.items()} if predictions and state.get("phase") != "blocked" else None,
              "remaining_time_calibration": {":".join(key): {"median_observed_to_pilot_ratio": value, "completed_samples": len(ratios[key])} for key,value in calibration.items()},
              "failure_signatures": dict(errors), "final_update": final}
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader"], capture_output=True, text=True)
    result["gpu_util_vram_temperature_power"] = gpu.stdout.strip() if gpu.returncode == 0 else "unavailable"
    text = "# Execution progress\n\n"
    text += f"UTC: {result['utc']}\n\nAsia/Seoul: {result['asia_seoul']}\n\nPhase: {result['phase']}; elapsed {result['elapsed_sec']/3600:.2f} hours.\n\n"
    text += f"Jobs: {json.dumps(result['counts'])}; total {len(plan)}; completed {result['percent']} percent.\n\n"
    text += f"Preceding 12 hours: {completed_recent} completions, {failed_recent} failures; observed throughput {result['observed_jobs_per_hour_last_12h']:.3f} jobs/hour.\n\n"
    text += f"Current jobs and elapsed seconds: {json.dumps(current)}\n\n"
    text += f"GPU utilization, used/total VRAM, temperature, power: {result['gpu_util_vram_temperature_power']}\n\n"
    text += f"CPU load: {result['load_average']}; RAM used {result['ram']['used']/2**30:.2f} GiB of {result['ram']['total']/2**30:.2f}; disk free {result['disk_free_bytes']/2**30:.2f} GiB.\n\n"
    text += f"Remaining seconds (low/central/high): {result['remaining_seconds']}\n\nCompletion in Asia/Seoul: {result['estimated_completion_asia_seoul']}\n\n"
    text += f"Blockers/failure signatures: {json.dumps(result['failure_signatures'])}\n\n"
    text += "Status: `./run_all.sh status`\n\nProgress: `sed -n '1,240p' source/results/progress/latest.md`\n\nAttach: `tmux attach -t korean-sentiment-all`\n\nResume validated stages: `conda run --no-capture-output -n ML python scripts/server_execution.py run`\n\nDetailed counts and forecast calibration:\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n"
    write(progress / "latest.md", text)
    write(progress / f"update_{now.strftime('%Y%m%dT%H%M%SZ')}.md", text)
    history = rows(progress / "history.csv")
    history.append({"utc": result["utc"], "phase": result["phase"], "elapsed_sec": result["elapsed_sec"], "total": len(plan), **result["counts"], "disk_free_bytes": result["disk_free_bytes"]})
    table(progress / "history.csv", history)
    with (progress / "monitor.log").open("a") as handle:
        handle.write(text + "\n")
    print(text, flush=True)


def run():
    from korean_sentiment.orchestration.aggregate import aggregate_all
    verify_plans(preview=True)
    assert (SERVER / "preparation_passed.json").exists(), "preparation gates missing"
    assert (SERVER / "runtime_estimate.json").exists(), "measured estimate missing"
    assert len(rows(SERVER / "smoke_matrix.csv")) == 12, "smoke matrix missing"
    assert all(r["status"] == "passed" for r in rows(SERVER / "smoke_matrix.csv")), "smoke failure"
    command([sys.executable, "-m", "pytest", "-q", "source/tests/test_server_forecast.py"], "forecast_tests")
    estimate()
    import fcntl
    lock_handle = (SERVER / "workflow.lock").open("a")
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopped = threading.Event()
    def monitor():
        snapshot()
        while not stopped.wait(43200):
            snapshot()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        for stage in ["tune", "full", "aggregate", "bonus-plan", "bonus-tune", "bonus-full", "bonus-aggregate"]:
            gate_disk()
            args = [str(ROOT / "run_all.sh"), stage]
            if stage in {"tune", "full", "bonus-tune", "bonus-full"}:
                args += ["--noninteractive-confirm-expensive", "--fail-fast"]
            command(args)
            aggregate_all()
            if stage in {"tune", "bonus-plan", "bonus-tune"}:
                estimate()
            if stage == "bonus-plan":
                verify_plans()
        phase("auditing")
    finally:
        stopped.set()
        thread.join(timeout=10)
        aggregate_all()
        if sys.exc_info()[0] is not None:
            phase("blocked")
        audited = audit_report()
        phase("completed" if audited and sys.exc_info()[0] is None else "blocked")
        snapshot(final=True)
        terminal_phase = read(SERVER / "workflow_state.json", {}).get("phase", "blocked")
        command([str(ROOT / "run_all.sh"), "archive-results"])
        archives = sorted((SOURCE / "results/archives").glob("korean_sentiment_results_*.tar.gz"))
        from korean_sentiment.utils.io import sha256_file
        if archives:
            path = archives[-1]
            write(SERVER / "archive_manifest.json", {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        phase(terminal_phase)
        lock_handle.close()
        if not audited and sys.exc_info()[0] is None:
            raise RuntimeError("completeness audit did not pass; see server/completeness_audit.csv")


def gate_disk():
    gate = read(SERVER / "storage_gate.json")
    baseline = gate["free_gib"] * 2**30
    peak = gate["peak_increment_estimate_gib"] * 2**30
    # Remaining capacity must cover the unconsumed estimated peak plus its 50% reserve.
    if shutil.disk_usage(ROOT).free < max(0, peak-(baseline-shutil.disk_usage(ROOT).free)) + 0.5*peak:
        raise RuntimeError("disk reserve gate failed; no prior artifacts were deleted")


def audit_report():
    from korean_sentiment.orchestration.state import validate_completed
    from korean_sentiment.utils.runtime import git_state
    output = []
    split_counts = {}
    identifiers = Counter(job["run_id"] for job in plans())
    for job in plans():
        directory = SOURCE / "results/run_records" / job["run_id"]
        status = read(directory / "status.json", {}).get("status", "missing")
        issues = []
        if identifiers[job["run_id"]] != 1:
            issues.append("duplicate planned run ID")
        if status == "completed":
            if not validate_completed(directory, job["data_manifest_hash"], job["config_hash"]):
                issues.append("invalid completed artifact")
            metrics = read(directory / "metrics.json", {})
            tuning = job["phase"].endswith("tuning")
            if tuning and "test" in metrics:
                issues.append("tuning contains test metrics")
            for kind in ["validation"] + ([] if tuning else ["test"]):
                m = metrics.get(kind, {})
                for key in ["macro_f1", "accuracy", "macro_precision", "macro_recall"]:
                    value = m.get(key)
                    if value is None or not math.isfinite(value) or not 0 <= value <= 1:
                        issues.append(f"invalid {kind}.{key}")
                cm = m.get("confusion_matrix", [])
                expected = int(sum(sum(r) for r in cm))
                from korean_sentiment.data.prepare import load_split
                key = (job["dataset"], kind)
                if key not in split_counts:
                    split_counts[key] = len(load_split(*key))
                if expected != split_counts[key]:
                    issues.append(f"{kind} confusion total mismatch")
        output.append({"run_id": job["run_id"], "phase": job["phase"], "status": status, "issues": "; ".join(issues)})
    table(SERVER / "completeness_audit.csv", output)
    counts = defaultdict(Counter)
    for row in output:
        counts[row["phase"]][row["status"]] += 1
    complete = bool(output) and all(r["status"] == "completed" and not r["issues"] for r in output)
    reference_issues = []
    completed_part1 = {r["run_id"] for r in output if r["phase"] == "part1_final" and r["status"] == "completed"}
    for r in rows(SOURCE / "results/csv/bonus_metrics_by_seed.csv"):
        if r.get("record_origin") == "reused_part1" and r["run_id"] not in completed_part1:
            reference_issues.append(r["run_id"])
    complete = complete and not reference_issues
    write(SERVER / "completeness_audit.md", "# Completeness audit\n\n" + json.dumps({"complete": complete, "counts": {k: dict(v) for k,v in counts.items()}, "invalid_bonus_reference_run_ids": reference_issues}, indent=2) + "\n\nPer-run issues: completeness_audit.csv.\n")
    state = read(SERVER / "workflow_state.json", {})
    text = "# Server execution report\n\n" + ("All planned training records passed the run-level audit." if complete else "Execution remains incomplete. No complete-project scientific conclusions are supported.")
    text += "\n\n```json\n" + json.dumps({"utc": stamp().isoformat(), **git_state(), "phase_counts": {k: dict(v) for k,v in counts.items()}, "elapsed_workflow_sec": time.time()-state.get("started_epoch", time.time())}, indent=2) + "\n```\n"
    text += "\nHardware: RTX 4070, 12282 MiB VRAM; driver535.230.02; i9-10900K (10 cores/20 threads), 31GiB RAM. All execution uses Conda ML; package exports and inventory are under source/results/server/. No concurrency is used. Detailed cleaning and split counts: preprocessing_report.md. Plans: source/results/plans/. Runtime pilot coverage and uncertainty: runtime_estimate.md/.json/.csv. Smoke/batch and VRAM measurements: smoke_matrix.csv. Status and missing/error rows: completeness_audit.csv. Monitor history: source/results/progress/history.csv. Stage logs: source/results/server/; job tracebacks: source/results/run_records/. No raw data or large model artifacts enter the results archive. Archive checksum: source/results/server/archive_manifest.json.\n"
    text += "\nAuthoritative CSVs: source/results/csv/run_records.csv, final_metrics_by_seed.csv, final_metrics_summary.csv, crossover_summary.csv, resource_summary.csv, bonus_metrics_by_seed.csv, bonus_metrics_summary.csv, bonus_label_efficiency.csv, bonus_representation_metrics.csv, complete_project_status.csv. These distinguish measured per-run outputs from derived summaries. Missing values remain missing. Three seeds do not establish statistical significance. Source-label savings count only new target-domain labels. No LaTeX or presentation figures are generated.\n"
    text += "\nResume: `conda run --no-capture-output -n ML python scripts/server_execution.py run` after fixing the recorded failure. Completed runs are skipped. Status: `./run_all.sh status`. Attach: `tmux attach -t korean-sentiment-all`.\n"
    if complete:
        def markdown(values, fields):
            return "\n\n| " + " | ".join(fields) + " |\n| " + " | ".join("---" for _ in fields) + " |\n" + "\n".join("| " + " | ".join(str(row.get(key, "")) for key in fields) + " |" for row in values) + "\n"
        csv_dir = SOURCE / "results/csv"
        summary = rows(csv_dir / "final_metrics_summary.csv")
        best = {}
        for row in summary:
            key = (row["dataset"], int(row["train_size"]))
            if key not in best or float(row["test_macro_f1_mean"]) > float(best[key]["test_macro_f1_mean"]):
                best[key] = row
        text += "\nPart 1 measured performance, best method at each size (descriptive selection by test mean):"
        text += markdown([best[key] for key in sorted(best)], ["dataset", "train_size", "model", "test_macro_f1_mean", "test_macro_f1_std"])
        crossover = rows(csv_dir / "crossover_summary.csv")
        text += "\nDerived crossover results:"
        text += markdown(list({r["dataset"]: r for r in crossover}.values()), ["dataset", "first_observed_crossover", "first_sustained_crossover"])
        full = {r["dataset"]: int(r["n_full"]) for r in rows(csv_dir / "dataset_statistics.csv")}
        text += "\nFull-data performance:"
        text += markdown([r for r in summary if int(r["train_size"]) == full[r["dataset"]]], ["dataset", "model", "train_size", "test_macro_f1_mean", "test_macro_f1_std"])
        text += "\nPart 2 measured target-domain macro-F1 and source retention:"
        text += markdown(rows(csv_dir / "bonus_metrics_summary.csv"), ["target_domain", "target_train_size", "method", "target_macro_f1_mean", "target_macro_f1_std", "mean_source_macro_f1_mean", "source_retention_difference_vs_pooled_ce"])
        text += "\nDerived target-label efficiency (unobserved thresholds remain missing):"
        text += markdown(rows(csv_dir / "bonus_label_efficiency.csv"), ["target_domain", "method", "crossover_threshold_size", "target_label_crossover_shift", "target_training_labels_saved"])
        records = [read(p) for p in (SOURCE / "results/run_records").glob("*/record.json")]
        records = [r for r in records if r.get("status") == "completed" and r.get("phase") != "smoke"]
        resources = []
        groups = defaultdict(list)
        for row in records:
            groups[(row["phase"], row["dataset"], row["model"], row["objective"])].append(row)
        for key, group in sorted(groups.items()):
            value = dict(zip(["phase", "dataset", "model", "objective"], key))
            for field in ["fit_time_sec", "classifier_latency_ms_p50", "end_to_end_latency_ms_p50", "serialized_size_mb", "peak_gpu_memory_mb", "cross_domain_sentiment_separation_gap"]:
                available = [float(r[field]) for r in group if r.get(field) not in (None, "")]
                value[field + "_mean"] = sum(available)/len(available) if available else None
            resources.append(value)
        table(SERVER / "cost_and_representation_summary.csv", resources)
        text += "\nMeasured training/inference cost and cosine-separation diagnostics are grouped by phase, dataset, model and objective in server/cost_and_representation_summary.csv; these are means across sizes and seeds, so use run-level resources for matched comparisons. Projection heads are excluded from deployed inference. Actual CPU process-hours and GPU utilization-hours are unavailable for production records; elapsed assigned-device hours must not be presented as those measurements. Peak storage was not continuously sampled and remains unavailable.\n"
        text += "\nInterpretation boundaries: noisy rating/recommendation labels, three specific domains, from-scratch representations, validation-selected configurations, finite tested label budgets and three seeds. Crossovers summarize observed means and do not demonstrate significance.\n"
        full_cost = summarize_costs([r for r in records if r.get("phase") == "part1_final" and int(r["train_size"]) == full[r["dataset"]]], ["dataset", "model", "train_size"])
        table(SERVER / "full_data_cost_summary.csv", full_cost)
        text += "\nFull-data costs (means of three seed measurements; latency columns average per-run medians, memory columns take the largest observed seed peak):"
        cost_fields = ["device", "fit_time_sec_mean", "classifier_latency_ms_p50_mean", "end_to_end_latency_ms_p50_mean", "serialized_size_mb_mean", "peak_host_ram_mb_max", "peak_gpu_memory_mb_max"]
        text += markdown(full_cost, ["dataset", "model", "train_size"] + cost_fields)
        record_map = {r["run_id"]: r for r in records}
        matched = []
        for reference in rows(csv_dir / "bonus_metrics_by_seed.csv"):
            source = record_map.get(reference["run_id"])
            if source:
                matched.append({**source, "target_domain": reference["target_domain"], "target_train_size": int(reference["target_train_size"]), "method": reference["method"]})
        bonus_cost = summarize_costs(matched, ["target_domain", "target_train_size", "method"])
        pooled = {(r["target_domain"], r["target_train_size"]): r for r in bonus_cost if r["method"] == "pooled_ce"}
        for r in bonus_cost:
            base = pooled.get((r["target_domain"], r["target_train_size"]), {}).get("fit_time_sec_mean")
            r["fit_time_ratio_vs_pooled_ce"] = r["fit_time_sec_mean"]/base if base is not None and base > 0 and r["fit_time_sec_mean"] is not None else None
        table(SERVER / "bonus_cost_summary.csv", bonus_cost)
        text += "\nPart 2 matched training and inference costs. Fit-time ratios include data preparation and early stopping, rather than measuring only the extra contrastive-loss operation. Serialized sizes are training artefacts; projection heads are bypassed for deployed inference. Device differences matter when comparing CPU TF-IDF and CUDA neural latency."
        text += markdown(bonus_cost, ["target_domain", "target_train_size", "method", "device", "fit_time_sec_mean", "fit_time_ratio_vs_pooled_ce", "classifier_latency_ms_p50_mean", "end_to_end_latency_ms_p50_mean", "deployed_parameter_count_mean"])
        geometry = [r for r in bonus_cost if r["method"] in {"pooled_ce", "supcon", "db_cdsupcon"}]
        table(SERVER / "bonus_cosine_summary.csv", geometry)
        text += "\nRepresentation cosine diagnostics, averaged over the three seeds using the fixed evaluation sample and pair seed:"
        text += markdown(geometry, ["target_domain", "target_train_size", "method", "same_sentiment_cross_domain_cosine_mean", "different_sentiment_cross_domain_cosine_mean", "cross_domain_sentiment_separation_gap_mean", "same_domain_same_sentiment_cosine_mean"])
        text += "\nFor complete curves, blank crossover thresholds mean the corresponding criterion was not observed within the tested sizes. They are not zero-label thresholds.\n"
    statistics_rows = rows(SOURCE / "results/csv/dataset_statistics.csv")
    text += "\nFinal dataset splits and actual training-size grids:\n\n```json\n" + json.dumps(statistics_rows, indent=2) + "\n```\n"
    preprocessing = SERVER / "preprocessing_report.md"
    if preprocessing.exists():
        text += "\nCleaning removals, class balance and manifests (authoritative preprocessing report):\n\n" + preprocessing.read_text().split("\n", 1)[-1]
    notes = SERVER / "numerical_notes.md"
    if notes.exists():
        text += "\nNumerical precision and timing boundaries:\n\n" + notes.read_text().split("\n", 1)[-1]
    write(SOURCE / "results/reports/SERVER_EXECUTION_REPORT.md", text)
    return complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "smoke", "probe", "pilot", "pilot-one", "estimate", "run", "snapshot", "audit"])
    parser.add_argument("--name")
    args = parser.parse_args()
    assert os.environ.get("CONDA_DEFAULT_ENV") == "ML", "must run inside Conda ML"
    SERVER.mkdir(parents=True, exist_ok=True)
    if not (SERVER / "workflow_state.json").exists():
        write(SERVER / "workflow_state.json", {"started_epoch": time.time(), "started_at": stamp().isoformat()})
    os.environ.setdefault("KS_PRECISION", "bf16")
    os.environ.setdefault("KS_CPU_THREADS", "4")
    # Set native pool limits before any NumPy/scikit-learn/PyTorch import.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = os.environ["KS_CPU_THREADS"]
    os.environ.setdefault("KS_NUM_WORKERS", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    gate = read(SERVER / "storage_gate.json", {})
    os.environ.setdefault("KS_MIN_FREE_GIB", str(gate.get("peak_increment_estimate_gib", 142) * 0.5))
    try:
        if args.action == "probe":
            memory_probe(args.name)
        elif args.action == "pilot-one":
            pilot_one(json.loads(args.name))
        else:
            {"prepare": prepare, "smoke": smoke, "pilot": pilot, "estimate": estimate, "run": run, "snapshot": snapshot, "audit": audit_report}[args.action]()
    except BaseException as error:
        write(SERVER / "workflow_error.json", {"utc": stamp().isoformat(), "error": str(error), "traceback": traceback.format_exc()})
        if args.action not in {"probe", "pilot-one"}:
            phase("blocked")
        raise


if __name__ == "__main__":
    main()
