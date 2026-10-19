"""Compute used per stage, from the scheduler logs -> ``results/aggregated/compute_usage.csv``.

A stage can span several scheduler invocations (segments: a resume after a restart or a fix).
Per segment, busy wall time runs from the first job start (finish time minus duration of each
``done`` line) to the last job end, so waits for exclusivity before the first job are excluded.
GPU utilisation is the mean of the 30 s ``nvidia-smi`` samples (logs/gpu_util_<stage>.csv)
inside the busy windows; CPU utilisation is the mean of the scheduler's progress samples
(percent of all logical cores).

Usage: ``python -m ksent.analysis.compute_usage``
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

import pandas as pd

from ksent.analysis.common import AGG_DIR
from ksent.config import SOURCE_DIR

STAGES = [  # (scheduler stage, label, run mode)
    ("pilot", "C pilot", "cost"),
    ("tuning_cost", "D tuning", "cost"),
    ("final_quality", "E main sweep", "quality"),
    ("ablation_quality", "F ablation", "quality"),
    ("cost_n8000", "G cost pass", "cost"),
    ("cost_crossover", "G crossover cost", "cost"),
]
_TS = r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]"
_DONE = re.compile(_TS + r" done\s+.*\[(gpu:\d+|cpu)\] ([\d.]+)s")
_PROG = re.compile(_TS + r" progress: .*\| CPU (\d+)%")
_START = re.compile(_TS + r" stage=")


def _t(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def _segments(lines: list[str]) -> list[list[str]]:
    segs: list[list[str]] = []
    for ln in lines:
        if _START.match(ln):
            segs.append([])
        if segs:
            segs[-1].append(ln)
    return segs


def _gpu_util(stage: str, windows: list[tuple[datetime, datetime]]) -> float:
    p = SOURCE_DIR / "logs" / f"gpu_util_{stage}.csv"
    if not p.exists() or not windows:
        return float("nan")
    df = pd.read_csv(p, skipinitialspace=True)
    df = df[df["timestamp"] != "timestamp"]          # header repeated by appended runs
    ts = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M:%S.%f")
    util = df["utilization.gpu [%]"].str.rstrip(" %").astype(float)
    mask = pd.Series(False, index=df.index)
    for a, b in windows:
        mask |= (ts >= a) & (ts <= b)
    return float(util[mask].mean()) if mask.any() else float("nan")


def stage_usage(stage: str, label: str, mode: str) -> dict | None:
    p = SOURCE_DIR / "logs" / f"scheduler_{stage}.log"
    if not p.exists():
        return None
    windows, runs, gpu_s, cpu_s, cpu_samples = [], 0, 0.0, 0.0, []
    for seg in _segments(p.read_text(encoding="utf-8").splitlines()):
        done = [m for m in map(_DONE.match, seg) if m]
        if not done:
            continue
        start = min(_t(m[1]) - timedelta(seconds=float(m[3])) for m in done)
        end = max(_t(m[1]) for m in done)
        windows.append((start, end))
        runs += len(done)
        gpu_s += sum(float(m[3]) for m in done if m[2].startswith("gpu"))
        cpu_s += sum(float(m[3]) for m in done if m[2] == "cpu")
        cpu_samples += [float(m[2]) for m in map(_PROG.match, seg) if m and start <= _t(m[1]) <= end]
    wall = sum((b - a).total_seconds() for a, b in windows)
    return {"stage": label, "scheduler_stage": stage, "run_mode": mode, "segments": len(windows), "runs": runs,
            "busy_wall_h": wall / 3600, "gpu_job_h": gpu_s / 3600, "cpu_job_h": cpu_s / 3600,
            "avg_gpu_util_pct": _gpu_util(stage, windows),
            "avg_cpu_util_pct": sum(cpu_samples) / len(cpu_samples) if cpu_samples else float("nan")}


def main() -> None:
    rows = [r for r in (stage_usage(*s) for s in STAGES) if r]
    df = pd.DataFrame(rows)
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(AGG_DIR / "compute_usage.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 2):
        print(df.drop(columns="scheduler_stage").to_string(index=False))
    print(f"  total busy wall: {df.busy_wall_h.sum():.2f} h  (1 GPU -> {df.busy_wall_h.sum():.2f} GPU-hours allocated)")


if __name__ == "__main__":
    main()
