"""Hardware-aware launcher: one :mod:`ksent.run_one` subprocess per job.

Two run modes (see :mod:`ksent.run_one`):

* ``quality`` - maximum utilisation. Neural jobs go to GPU lanes (one lane per visible
  GPU, ``K_gpu`` concurrent jobs each, pinned with ``CUDA_VISIBLE_DEVICES``); classical
  jobs go to a CPU lane of ``cpu_jobs`` concurrent jobs with ``n_threads`` threads each.
  Timings are marked not reportable.
* ``cost`` - one job at a time, and only while the machine is otherwise idle: before
  every job the scheduler waits until no foreign GPU compute process is running and
  the CPU load is low (``--no-exclusive-check`` disables the wait, for tests only).

Concurrency never changes a job's thread count or device, so predictions do not depend
on the mode. Finished results are skipped, so a stage can be restarted at any time.

Failures: every failed job is retried once; a CUDA out-of-memory failure is retried
alone on its GPU. A job that still fails (crash, NaN loss) is appended to
``results/failed.jsonl`` and the stage continues.

Progress (running/finished/failed, GPU and CPU utilisation, ETA) is printed every
``--progress-every`` seconds and appended to ``logs/scheduler_<stage>.log``;
``nvidia-smi`` samples GPU utilisation every 30 s into ``logs/gpu_util_<stage>.csv``.

Examples::

    python -m ksent.scheduler --kind pilot --run-mode cost --sizes 2000 8000
    python -m ksent.scheduler --kind tuning --run-mode cost
    python -m ksent.scheduler --kind final --run-mode quality
    python -m ksent.scheduler --kind final --run-mode cost --sizes 8000
    python -m ksent.scheduler --kind final --run-mode quality --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import psutil

from ksent.config import SOURCE_DIR, family, global_cfg, path, resolve_size
from ksent.run_one import default_out_root, is_done, result_path
from ksent.sweep import Job, enumerate_jobs
from ksent.utils import now_iso, read_json

# relative cost of a method at equal size, used for longest-first ordering when no pilot exists
_WEIGHT = {"tfidf_lr": 1, "tfidf_svm": 1, "tfidf_rf": 8, "tfidf_xgb": 8, "emb_mlp": 2, "emb_rnn_uni": 2, "emb_lstm_uni": 3,
           "emb_gru_uni": 3, "emb_rnn": 3, "emb_lstm": 4,
           "emb_gru": 4, "small_transformer": 6, "small_mamba": 8}


# ----------------------------------------------------------------------------- hardware probes
def gpu_stats() -> list[dict]:
    """Per-GPU utilisation and memory from nvidia-smi (empty list without NVIDIA GPUs)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        i, u, used, tot = (x.strip() for x in line.split(","))
        rows.append({"index": int(i), "util": float(u), "used_mb": float(used), "total_mb": float(tot)})
    return rows


def foreign_gpu_pids(own: set[int]) -> list[int]:
    """GPU compute processes that are not ours (or our children)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    pids = [int(x) for x in out.split() if x.strip().isdigit()]
    return [p for p in pids if p not in own]


def foreign_cpu_load(own: set[int]) -> float:
    """CPU cores busy with processes that are not ours, sampled over 1 s."""
    procs = []
    for p in psutil.process_iter(["pid"]):
        if p.info["pid"] in own or p.info["pid"] == os.getpid():
            continue
        try:
            p.cpu_percent(None)
            procs.append(p)
        except psutil.Error:
            pass
    time.sleep(1.0)
    busy = 0.0
    for p in procs:
        try:
            busy += p.cpu_percent(None)
        except psutil.Error:
            pass
    return busy / 100.0


def physical_cores() -> int:
    return psutil.cpu_count(logical=False) or os.cpu_count() or 1


def visible_gpus() -> list[str]:
    """GPU ids to schedule on: ``CUDA_VISIBLE_DEVICES`` if set, else every GPU nvidia-smi lists."""
    env = os.environ.get("KSENT_GPUS") or os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [x.strip() for x in env.split(",") if x.strip()]
    return [str(g["index"]) for g in gpu_stats()]


# ----------------------------------------------------------------------------- jobs
@dataclass
class Task:
    job: Job
    cost: float
    attempts: int = 0
    exclusive: bool = False                      # rerun alone on its GPU (after CUDA OOM)
    errors: list[str] = field(default_factory=list)


def _estimates() -> dict:
    """Pilot-derived per-(method, size) estimates: fit seconds, peak RSS and VRAM (MB)."""
    p = SOURCE_DIR / "configs" / "scheduler_estimates.json"
    return read_json(p) if p.exists() else {}


def _lookup(est: dict, method: str, size: int, key: str, default: float) -> float:
    """Interpolate an estimate linearly in size from the measured pilot sizes."""
    pts = sorted((int(n), v[key]) for n, v in (est.get(method) or {}).items() if v.get(key) is not None)
    if not pts:
        return default
    if len(pts) == 1:
        n0, v0 = pts[0]
        return v0 * size / n0 if key == "fit_s" else v0
    for (n0, v0), (n1, v1) in zip(pts, pts[1:]):
        if size <= n1 or (n1, v1) == pts[-1]:
            return max(0.0, v0 + (v1 - v0) * (size - n0) / (n1 - n0))
    return pts[-1][1]


def job_cost(est: dict, j: Job) -> float:
    base = _lookup(est, j.method, j.size, "fit_s", -1.0)
    if base < 0:
        base = _WEIGHT.get(j.method, 4) * j.size / 1000
    return base * (1.5 if j.tokenizer == "char" and family(j.method) == "neural" else 1.0)


# ----------------------------------------------------------------------------- scheduler
class Scheduler:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        g = global_cfg()
        sc = g.get("scheduler") or {}
        self.n_threads = int(os.environ.get("KSENT_THREADS") or g["n_threads"])
        self.neural_threads = int(g.get("neural_threads") or self.n_threads)
        self.gpus = visible_gpus() if args.device != "cpu" else []
        self.k_gpu = int(args.k_gpu or sc.get("k_gpu") or 1)
        reserved = self.k_gpu * len(self.gpus) * self.neural_threads + int(sc.get("reserved_system_cores", 2))
        default_cpu_jobs = max(1, (physical_cores() - reserved) // self.n_threads)
        self.cpu_jobs = int(args.cpu_jobs or sc.get("cpu_jobs") or default_cpu_jobs)
        if args.run_mode == "cost":
            self.k_gpu, self.cpu_jobs = 1, 1
        total_threads = self.cpu_jobs * self.n_threads + self.k_gpu * len(self.gpus) * self.neural_threads
        if args.run_mode == "quality" and total_threads > physical_cores():
            raise SystemExit(f"oversubscribed: {total_threads} threads > {physical_cores()} physical cores")
        self.vram_frac = float(sc.get("vram_max_fraction", 0.85))
        self.ram_margin_mb = float(sc.get("ram_margin_mb", 4096))
        self.est = _estimates()
        self.stage = args.stage or f"{args.kind}_{args.run_mode}"
        self.run_kind = "final" if args.kind == "ablation" else args.kind
        self.out_root = Path(args.out_root) if args.out_root else default_out_root(self.run_kind, None, args.run_mode)
        self.log_path = SOURCE_DIR / "logs" / f"scheduler_{self.stage}.log"
        self.run_log_dir = SOURCE_DIR / "logs" / "runs" / self.stage
        self.failed_path = path("results") / "failed.jsonl"
        self.running: dict[int, tuple[Task, subprocess.Popen, float, str]] = {}
        self.finished: list[tuple[Task, float]] = []
        self.failed: list[Task] = []
        self.util_samples: list[tuple[float, float]] = []   # (gpu util mean, cpu util %)

    # ---------------------------------------------------------------- helpers
    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def rpath(self, j: Job) -> Path:
        return result_path(self.out_root, self.run_kind, j.dataset, j.method, j.tokenizer, j.size, j.seed, j.config_id)

    def done(self, j: Job) -> bool:
        return is_done(self.rpath(j), {"dataset": j.dataset, "method": j.method, "seed": j.seed})

    def cmd(self, t: Task) -> list[str]:
        j, a = t.job, self.args
        c = [sys.executable, "-m", "ksent.run_one", "--mode", self.run_kind, "--run-mode", a.run_mode,
             "--dataset", j.dataset, "--method", j.method, "--tokenizer", j.tokenizer, "--size", str(j.size),
             "--seed", str(j.seed), "--out-root", str(self.out_root)]
        if j.config_id is not None:
            c += ["--config-id", str(j.config_id)]
        if family(j.method) == "neural":
            c += ["--device", "cuda" if self.gpus else (a.device or "cpu")]
        if a.mamba_backend:
            c += ["--mamba-backend", a.mamba_backend]
        if a.max_epochs:
            c += ["--max-epochs", str(a.max_epochs)]
        if a.allow_fallback:
            c.append("--allow-fallback")
        if a.kind in ("smoke", "pilot") and a.run_mode == "quality":
            c.append("--not-reportable")
        return c

    def lanes_free(self, t: Task) -> str | None:
        """Lane to launch ``t`` on now, or None (capacity, exclusivity and memory guard)."""
        running = list(self.running.values())
        if self.args.run_mode == "cost":
            return ("gpu:" + self.gpus[0] if self.gpus and family(t.job.method) == "neural" else "cpu") \
                if not running else None
        if family(t.job.method) == "tfidf" or not self.gpus:
            if sum(1 for r in running if r[3] == "cpu") >= self.cpu_jobs:
                return None
            need = _lookup(self.est, t.job.method, t.job.size, "peak_rss_mb", 2048)
            return "cpu" if self._ram_ok(need) else None
        # capacity first: skip the nvidia-smi query when every GPU lane is full
        if all(sum(1 for r in running if r[3] == f"gpu:{g}") >= self.k_gpu for g in self.gpus):
            return None
        stats = {str(g["index"]): g for g in self._gpu_stats()}
        need_vram = _lookup(self.est, t.job.method, t.job.size, "peak_vram_mb", 1024)
        for gid in sorted(self.gpus, key=lambda g: sum(1 for r in running if r[3] == f"gpu:{g}")):
            on = [r for r in running if r[3] == f"gpu:{gid}"]
            if any(r[0].exclusive for r in on) or (t.exclusive and on) or len(on) >= self.k_gpu:
                continue
            s = stats.get(gid)
            # reserve the estimated peak of jobs on this GPU that may not have allocated yet
            pending = sum(_lookup(self.est, r[0].job.method, r[0].job.size, "peak_vram_mb", 1024) for r in on)
            if s and on and s["used_mb"] + max(need_vram, 0) + pending * 0.5 > self.vram_frac * s["total_mb"]:
                continue
            if self._ram_ok(_lookup(self.est, t.job.method, t.job.size, "peak_rss_mb", 3072)):
                return f"gpu:{gid}"
        return None

    def _gpu_stats(self, max_age: float = 2.0) -> list[dict]:
        """gpu_stats() cached for ``max_age`` seconds (one query per scheduling pass, not per queued job)."""
        if time.time() - self._gpu_stats_ts > max_age:
            self._gpu_stats_cache, self._gpu_stats_ts = gpu_stats(), time.time()
        return self._gpu_stats_cache

    def _ram_ok(self, need_mb: float) -> bool:
        if not self.running:  # estimates are extrapolated above the pilot sizes: never block an idle machine
            return True
        reserved = sum(_lookup(self.est, r[0].job.method, r[0].job.size, "peak_rss_mb", 2048)
                       for r in self.running.values()) * 0.5
        return psutil.virtual_memory().available / 2**20 - reserved - need_mb > self.ram_margin_mb

    def wait_exclusive(self) -> None:
        """Cost mode: block until no foreign GPU process runs and foreign CPU load is low."""
        if self.args.no_exclusive_check:
            return
        own = {os.getpid()} | {c.pid for c in psutil.Process().children(recursive=True)}
        waited, last = 0.0, -1e9
        while True:
            fg = foreign_gpu_pids(own) if self.gpus else []
            load = foreign_cpu_load(own)
            if not fg and load <= self.args.max_foreign_load:
                if waited:
                    self.log(f"machine idle again after waiting {waited / 60:.1f} min; resuming cost runs")
                return
            if waited - last >= 600:
                self.log(f"cost mode: waiting for exclusivity (foreign GPU pids={fg}, foreign CPU load={load:.1f} "
                         f"cores > {self.args.max_foreign_load})")
                last = waited
            time.sleep(30)
            waited += 31

    def launch(self, t: Task, lane: str) -> None:
        env = {**os.environ, "PYTHONPATH": str(SOURCE_DIR / "src")}
        if lane.startswith("gpu:"):
            env["CUDA_VISIBLE_DEVICES"] = lane.split(":", 1)[1]
        t.attempts += 1
        self.run_log_dir.mkdir(parents=True, exist_ok=True)
        name = t.job.label().replace(" ", "_").replace("=", "")
        fh = open(self.run_log_dir / f"{name}.a{t.attempts}.log", "w")
        p = subprocess.Popen(self.cmd(t), cwd=SOURCE_DIR, env=env, stdout=fh, stderr=subprocess.STDOUT)
        self.running[p.pid] = (t, p, time.time(), lane)

    def failure_text(self, t: Task) -> str:
        fp = self.rpath(t.job).with_suffix(".failed.json")
        try:
            return read_json(fp).get("error", "") if fp.exists() else ""
        except Exception:
            return ""

    def record_failure(self, t: Task) -> None:
        self.failed.append(t)
        rec = {"timestamp": now_iso(), "stage": self.stage, "run_mode": self.args.run_mode, "kind": self.args.kind,
               "dataset": t.job.dataset, "method": t.job.method, "tokenizer": t.job.tokenizer, "size": t.job.size,
               "seed": t.job.seed, "config_id": t.job.config_id, "attempts": t.attempts, "errors": t.errors,
               "logs": str(self.run_log_dir.relative_to(SOURCE_DIR))}
        self.failed_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.failed_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def progress(self, n_total: int, n_queue: int, t0: float) -> None:
        gs = gpu_stats()
        gutil = sum(g["util"] for g in gs) / len(gs) if gs else float("nan")
        gmem = ", ".join(f"gpu{g['index']} {g['util']:.0f}% {g['used_mb'] / 1024:.1f}GB" for g in gs)
        cpu = psutil.cpu_percent(interval=None)
        self.util_samples.append((gutil, cpu))
        done = len(self.finished)
        el = time.time() - t0
        # ETA from throughput in estimated-cost units (long jobs run first)
        done_cost = sum(t.cost for t, _ in self.finished) or 1e-9
        left_cost = sum(t.cost for t, *_ in self.running.values()) * 0.5 + self._queue_cost
        eta = el * left_cost / done_cost if done else float("nan")
        self.log(f"progress: running={len(self.running)} finished={done} failed={len(self.failed)} "
                 f"queued={n_queue} total={n_total} | {gmem or 'no GPU'} | CPU {cpu:.0f}% | elapsed {_fmt(el)} "
                 f"| ETA {_fmt(eta)}")

    # ---------------------------------------------------------------- main loop
    def run(self, jobs: list[Job]) -> int:
        todo = [j for j in jobs if not self.done(j)]
        tasks = sorted((Task(j, job_cost(self.est, j)) for j in todo), key=lambda t: -t.cost)
        self.log(f"stage={self.stage} kind={self.args.kind} run_mode={self.args.run_mode} total={len(jobs)} "
                 f"done={len(jobs) - len(todo)} todo={len(todo)} out={self.out_root} | gpus={self.gpus} "
                 f"k_gpu={self.k_gpu} cpu_jobs={self.cpu_jobs} n_threads={self.n_threads} "
                 f"neural_threads={self.neural_threads} physical_cores={physical_cores()}")
        if self.args.dry_run:
            for t in tasks:
                print(f"   {t.cost:9.1f}  {t.job.label()}")
            return 0
        if not tasks:
            return 0
        smi = None
        if self.gpus:
            fh = open(SOURCE_DIR / "logs" / f"gpu_util_{self.stage}.csv", "a")
            smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used",
                                    "--format=csv", "-l", "30"], stdout=fh, stderr=subprocess.DEVNULL)
        psutil.cpu_percent(interval=None)
        queue, t0, last_prog = list(tasks), time.time(), 0.0
        try:
            while queue or self.running:
                self._queue_cost = sum(t.cost for t in queue)
                launched = True
                while queue and launched:
                    launched = False
                    for t in list(queue):
                        if self.args.run_mode == "cost" and not self.running:
                            self.wait_exclusive()
                        lane = self.lanes_free(t)
                        if lane:
                            queue.remove(t)
                            self.launch(t, lane)
                            launched = True
                            break
                        if self.args.run_mode == "cost":
                            break
                for pid, (t, p, ts, lane) in list(self.running.items()):
                    if p.poll() is None:
                        continue
                    del self.running[pid]
                    dur = time.time() - ts
                    if p.returncode == 0 and self.done(t.job):
                        self.finished.append((t, dur))
                        self.log(f"done   {t.job.label()} [{lane}] {dur:.1f}s (attempt {t.attempts})")
                        continue
                    err = self.failure_text(t) or f"exit code {p.returncode}"
                    t.errors.append(err[:500])
                    oom = "out of memory" in err.lower() or "outofmemory" in err.lower()
                    if t.attempts < 2:
                        t.exclusive = t.exclusive or oom
                        queue.insert(0, t)
                        self.log(f"retry  {t.job.label()} [{lane}] ({'CUDA OOM -> alone on GPU' if oom else err[:120]})")
                    else:
                        self.record_failure(t)
                        self.log(f"FAILED {t.job.label()} [{lane}] after {t.attempts} attempts: {err[:200]}")
                if time.time() - last_prog >= self.args.progress_every:
                    self.progress(len(tasks), len(queue), t0)
                    last_prog = time.time()
                time.sleep(0.5)
        finally:
            if smi:
                smi.terminate()
        self.progress(len(tasks), 0, t0)
        if self.util_samples:
            import math
            g = [u for u, _ in self.util_samples if not math.isnan(u)]
            c = [c for _, c in self.util_samples]
            self.log(f"stage summary: avg GPU util {sum(g) / len(g) if g else float('nan'):.1f}% "
                     f"avg CPU util {sum(c) / len(c):.1f}% over {len(self.util_samples)} samples")
        self.log(f"finished {len(self.finished)} runs in {_fmt(time.time() - t0)}; failed={len(self.failed)}")
        for t in self.failed:
            self.log(f"   FAILED: {t.job.label()} -> results/failed.jsonl")
        return 1 if self.failed else 0

    _queue_cost = 0.0
    _gpu_stats_cache: list[dict] = []
    _gpu_stats_ts = 0.0


def _fmt(sec: float) -> str:
    if sec != sec:  # NaN
        return "?"
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{s:02d}s"


def build_jobs(args: argparse.Namespace) -> list[Job]:
    g = global_cfg()
    datasets, methods = args.dataset or g["datasets"], args.method or g["methods"]
    seeds = args.seeds or g["seeds"]
    if args.kind == "pilot":
        sizes = args.sizes or [int(g["tuning_size"])]
        return [Job(args.pilot_dataset, m, g["tokenizer"], resolve_size(args.pilot_dataset, n), int(g["tuning_seed"]))
                for m in methods for n in sizes]
    if args.kind == "crossover_cost":
        from ksent.analysis.crossover import crossover_cost_jobs

        args.kind = "final"
        return [j for j in crossover_cost_jobs(seeds) if j.dataset in datasets]
    if args.kind == "ablation":
        return enumerate_jobs("ablation", datasets, methods, g["sizes"], seeds)
    return enumerate_jobs(args.kind, datasets, methods, args.sizes or g["sizes"], seeds)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=["final", "tuning", "ablation", "smoke", "pilot", "crossover_cost"])
    ap.add_argument("--run-mode", required=True, choices=["quality", "cost"])
    ap.add_argument("--stage", default=None, help="name for logs (default: <kind>_<run-mode>)")
    ap.add_argument("--dataset", nargs="*", default=None)
    ap.add_argument("--method", nargs="*", default=None)
    ap.add_argument("--sizes", nargs="*", default=None, help="training sizes (integers or 'full')")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--pilot-dataset", default="nsmc")
    ap.add_argument("--device", default=None, help="cpu forces neural jobs onto the CPU (smoke test)")
    ap.add_argument("--k-gpu", type=int, default=None)
    ap.add_argument("--cpu-jobs", type=int, default=None)
    ap.add_argument("--mamba-backend", default=None)
    ap.add_argument("--max-epochs", type=int, default=None, help="smoke only")
    ap.add_argument("--allow-fallback", action="store_true", help="smoke only")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--progress-every", type=float, default=180)
    ap.add_argument("--max-foreign-load", type=float, default=2.0, help="cost mode: max foreign CPU cores busy")
    ap.add_argument("--no-exclusive-check", action="store_true", help="cost mode without the idle wait (tests)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.max_epochs and args.kind not in ("smoke",):
        ap.error("--max-epochs is for smoke tests only")
    jobs = build_jobs(args)
    return Scheduler(args).run(jobs)


if __name__ == "__main__":
    raise SystemExit(main())
