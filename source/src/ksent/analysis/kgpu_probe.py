"""Choose ``K_gpu`` (concurrent neural jobs per GPU) for the quality-mode scheduler.

For K in ``--ks`` the probe launches K concurrent neural pilot jobs (n=8,000, pilot
dataset, tuning seed, default configs; the five neural methods cycled) on GPU 0 and
compares each job's training seconds per epoch with the same method run alone (the
exclusive n=8,000 pilot in ``results_pilot/``). It samples GPU memory with nvidia-smi.

Selected K = the largest K whose mean per-job slowdown is <= ``--max-slowdown`` (1.2)
and whose peak VRAM stays below ``scheduler.vram_max_fraction`` (0.85) of capacity.
Like cost mode, the probe waits until no foreign GPU process is running.

Output: ``results_pilot/kgpu_probe.csv`` and ``results_pilot/kgpu_choice.json``.
Usage: ``python -m ksent.analysis.kgpu_probe [--ks 2 3 4 6 8]``
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time

import pandas as pd

from ksent.config import NEURAL_METHODS, SOURCE_DIR, global_cfg
from ksent.scheduler import foreign_gpu_pids, gpu_stats
from ksent.utils import read_json, write_json

ROOT = SOURCE_DIR / "results_pilot"


def _sec_per_epoch(p) -> float | None:
    if not p.exists():
        return None
    r = read_json(p)
    if r.get("status") != "ok":
        return None
    return r["fit"]["classifier_time_s"] / max(1, r["fit"]["epochs_run"])


def main() -> int:
    g = global_cfg()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ks", nargs="*", type=int, default=[2, 3, 4, 6, 8])
    ap.add_argument("--dataset", default="nsmc")
    ap.add_argument("--size", type=int, default=8000)
    ap.add_argument("--max-slowdown", type=float, default=1.2)
    args = ap.parse_args()
    seed, tok = int(g["tuning_seed"]), g["tokenizer"]
    frac = float((g.get("scheduler") or {}).get("vram_max_fraction", 0.85))
    alone = {m: _sec_per_epoch(ROOT / "raw" / args.dataset / m / tok / f"n{args.size}_s{seed}.json")
             for m in NEURAL_METHODS}
    if any(v is None for v in alone.values()):
        raise SystemExit(f"exclusive n={args.size} pilot runs missing: {alone}")
    env = {**os.environ, "PYTHONPATH": str(SOURCE_DIR / "src"), "CUDA_VISIBLE_DEVICES": "0"}
    rows = []
    for k in args.ks:
        while foreign_gpu_pids({os.getpid()}):
            print("kgpu probe: waiting for an idle GPU ...", flush=True)
            time.sleep(60)
        base = gpu_stats()[0]["used_mb"]
        out = ROOT / "kgpu" / f"K{k}"
        procs = []
        for i in range(k):
            m = NEURAL_METHODS[i % len(NEURAL_METHODS)]
            # distinct output roots so that repeated methods do not collide
            o = out / f"job{i}"
            cmd = [sys.executable, "-m", "ksent.run_one", "--mode", "pilot", "--run-mode", "quality", "--dataset",
                   args.dataset, "--method", m, "--tokenizer", tok, "--size", str(args.size), "--seed", str(seed),
                   "--device", "cuda", "--out-root", str(o), "--force"]
            o.mkdir(parents=True, exist_ok=True)
            procs.append((m, o, subprocess.Popen(cmd, cwd=SOURCE_DIR, env=env, stdout=open(o / "log.txt", "w"),
                                                 stderr=subprocess.STDOUT)))
        peak, util, stop = [base], [], threading.Event()

        def sample() -> None:
            while not stop.is_set():
                s = gpu_stats()[0]
                peak.append(s["used_mb"])
                util.append(s["util"])
                time.sleep(2)

        th = threading.Thread(target=sample, daemon=True)
        th.start()
        t0 = time.time()
        for _, _, p in procs:
            p.wait()
        stop.set()
        th.join()
        wall = time.time() - t0
        total = gpu_stats()[0]["total_mb"]
        slow = []
        for m, o, p in procs:
            spe = _sec_per_epoch(o / "raw" / args.dataset / m / tok / f"n{args.size}_s{seed}.json")
            slow.append(spe / alone[m] if spe else float("nan"))
        row = {"k": k, "mean_slowdown": float(pd.Series(slow).mean()), "max_slowdown": float(pd.Series(slow).max()),
               "failed": sum(p.returncode != 0 for _, _, p in procs), "peak_vram_mb": max(peak),
               "peak_vram_frac": max(peak) / total, "mean_gpu_util": float(pd.Series(util).mean()) if util else None,
               "wall_s": wall, "jobs_per_hour": k / wall * 3600}
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "kgpu_probe.csv", index=False)
    ok = df[(df.failed == 0) & (df.mean_slowdown <= args.max_slowdown) & (df.peak_vram_frac < frac)]
    choice = int(ok.k.max()) if len(ok) else 1
    write_json({"k_gpu": choice, "max_slowdown": args.max_slowdown, "vram_max_fraction": frac,
                "alone_sec_per_epoch": alone, "probe": rows}, ROOT / "kgpu_choice.json")
    print(df.round(3).to_string(index=False))
    print(f"chosen k_gpu = {choice}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
