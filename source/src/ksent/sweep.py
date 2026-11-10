"""Launch many :mod:`ksent.run_one` runs, each in its own subprocess.

Runs are sequential by default (required for reportable timings). ``--parallel K``
runs K at a time and marks every result ``timing_reportable: false``; it is only
accepted for tuning and smoke tests. Completed runs (valid JSON with
``status == "ok"``) are skipped.

Modes: ``final`` (675 main fits, main tokenizer), ``tuning`` (270), ``ablation``
(tokenizer ablation; the 60 main-tokenizer cells already exist -> 120 new fits),
``smoke`` and ``pilot``.

Examples::

    python -m ksent.sweep --mode final
    python -m ksent.sweep --mode final --dataset nsmc --method tfidf_lr emb_gru --seeds 13 42
    python -m ksent.sweep --mode tuning --parallel 2
    python -m ksent.sweep --mode ablation
    python -m ksent.sweep --mode smoke --device cpu --max-epochs 2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass

from ksent.config import SOURCE_DIR, family, global_cfg, main_tokenizer, resolve_sizes, search_space
from ksent.run_one import default_out_root, is_done, result_path


@dataclass(frozen=True)
class Job:
    dataset: str
    method: str
    tokenizer: str
    size: int
    seed: int
    config_id: int | None = None

    def label(self) -> str:
        cfg = f" cfg={self.config_id}" if self.config_id is not None else ""
        return f"{self.dataset} {self.method} {self.tokenizer} n={self.size} seed={self.seed}{cfg}"


def enumerate_jobs(mode: str, datasets: list[str], methods: list[str], sizes: list, seeds: list[int],
                   pilot_dataset: str = "nsmc") -> list[Job]:
    """All jobs of ``mode`` in loop order dataset -> method -> (tokenizer) -> size -> seed.
    ``sizes`` may contain ``"full"``; it is resolved per dataset."""
    g, tok = global_cfg(), main_tokenizer()
    if mode == "tuning":
        return [Job(d, m, tok, int(g["tuning_size"]), int(g["tuning_seed"]), int(c["id"]))
                for d in datasets for m in methods for c in search_space(m)["configs"]]
    if mode == "smoke":
        return [Job(d, m, tok, resolve_sizes(d)[0], int(g["seeds"][0])) for d in datasets for m in methods]
    if mode == "pilot":
        return [Job(pilot_dataset, m, tok, int(g["tuning_size"]), int(g["tuning_seed"])) for m in methods]
    if mode == "ablation":
        from ksent.analysis.ablation import ablation_methods

        chosen = ablation_methods()
        ab = g["ablation"]
        return [Job(d, m, t, int(n), int(s)) for d in datasets for m in chosen.get(d, []) if m in methods
                for t in ab["tokenizers"] for n in ab["sizes"] for s in seeds]
    return [Job(d, m, tok, n, s) for d in datasets for m in methods for n in resolve_sizes(d, sizes) for s in seeds]


def _cmd(job: Job, mode: str, extra: list[str]) -> list[str]:
    run_mode = "final" if mode == "ablation" else mode
    cmd = [sys.executable, "-m", "ksent.run_one", "--mode", run_mode, "--dataset", job.dataset, "--method", job.method,
           "--tokenizer", job.tokenizer, "--size", str(job.size), "--seed", str(job.seed)]
    if job.config_id is not None:
        cmd += ["--config-id", str(job.config_id)]
    return cmd + extra


def _fmt(sec: float) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{s:02d}s"


def main() -> int:
    g = global_cfg()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["final", "tuning", "ablation", "smoke", "pilot"], default="final")
    ap.add_argument("--dataset", nargs="*", default=None)
    ap.add_argument("--method", nargs="*", default=None)
    ap.add_argument("--sizes", nargs="*", default=None, help="training sizes (integers or 'full')")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--pilot-dataset", default="nsmc")
    ap.add_argument("--device", default=None, help="passed to every neural run (default: config/KSENT_DEVICE)")
    ap.add_argument("--allow-fallback", action="store_true", help="smoke tests only")
    ap.add_argument("--mamba-backend", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--parallel", type=int, default=1, help=">1 marks timings not reportable (tuning/smoke only)")
    ap.add_argument("--max-epochs", type=int, default=None, help="smoke only")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.parallel > 1 and args.mode not in ("tuning", "smoke"):
        ap.error("--parallel is only allowed for tuning and smoke tests (timing fairness)")

    from ksent.hardware import configured_device, resolve_device

    device_spec = args.device or configured_device()
    neural_dev, why = resolve_device(device_spec, allow_fallback=args.allow_fallback)
    print(f"[sweep] neural device: {neural_dev} ({why})")
    jobs = enumerate_jobs(args.mode, args.dataset or g["datasets"], args.method or g["methods"],
                          args.sizes or g["sizes"], args.seeds or g["seeds"], args.pilot_dataset)
    out_root = default_out_root("final" if args.mode == "ablation" else args.mode, neural_dev)
    if args.out_root:
        from pathlib import Path

        out_root = Path(args.out_root)
    run_mode = "final" if args.mode == "ablation" else args.mode

    def done(j: Job) -> bool:
        p = result_path(out_root, run_mode, j.dataset, j.method, j.tokenizer, j.size, j.seed, j.config_id)
        return is_done(p, {"dataset": j.dataset, "method": j.method, "seed": j.seed})

    todo = [j for j in jobs if not done(j)]
    print(f"[sweep] mode={args.mode} total={len(jobs)} done={len(jobs) - len(todo)} todo={len(todo)} "
          f"parallel={args.parallel} out={out_root}")
    if args.dry_run:
        for j in todo:
            print("  ", j.label())
        return 0
    extra = ["--out-root", str(out_root), "--device", device_spec]
    if args.parallel > 1:
        extra.append("--not-reportable")
    if args.max_epochs:
        extra += ["--max-epochs", str(args.max_epochs)]
    if args.allow_fallback:
        extra.append("--allow-fallback")
    if args.mamba_backend:
        extra += ["--mamba-backend", args.mamba_backend]
    env = {**os.environ, "PYTHONPATH": str(SOURCE_DIR / "src")}
    t0, failures, finished = time.perf_counter(), [], 0
    offset = len(jobs) - len(todo)
    running: list[tuple[Job, subprocess.Popen, float]] = []
    queue = list(todo)
    log_dir = SOURCE_DIR / "logs" / "runs" / args.mode
    log_dir.mkdir(parents=True, exist_ok=True)
    while queue or running:
        while queue and len(running) < max(1, args.parallel):
            job = queue.pop(0)
            fh = open(log_dir / f"{job.label().replace(' ', '_').replace('=', '')}.log", "w")
            p = subprocess.Popen(_cmd(job, args.mode, extra), cwd=SOURCE_DIR, env=env, stdout=fh,
                                 stderr=subprocess.STDOUT)
            running.append((job, p, time.perf_counter()))
        for item in list(running):
            job, p, ts = item
            if p.poll() is None:
                continue
            running.remove(item)
            finished += 1
            dur = time.perf_counter() - ts
            elapsed = time.perf_counter() - t0
            eta = elapsed / finished * (len(todo) - finished)
            dev = "cpu" if family(job.method) == "tfidf" else neural_dev
            status = "done" if p.returncode == 0 else "FAILED"
            if p.returncode != 0:
                failures.append(job)
            print(f"[{offset + finished}/{len(jobs)}] {job.label()} [{dev}] {status} {dur:.1f}s | "
                  f"elapsed {_fmt(elapsed)} | ETA {_fmt(eta)}", flush=True)
        time.sleep(0.05)
    print(f"[sweep] finished {finished} runs in {_fmt(time.perf_counter() - t0)}; failures={len(failures)}")
    for j in failures:
        print("   FAILED:", j.label(), f"(log: logs/runs/{args.mode}/)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
