"""Run one experiment: ``(dataset, method, tokenizer, size, seed[, config_id])`` -> one JSON.

Each run executes in its own subprocess (see :mod:`ksent.sweep`) so the peak-RSS
measurement belongs to that run only.

Modes
  final   frozen config, test evaluation, latency/memory/size, predictions CSV
          -> results/raw/{d}/{m}/{tok}/n{size}_s{seed}.json   (ablation runs = final runs
             with a non-main tokenizer)
  tuning  ``--config-id k`` at the tuning size/seed with the main tokenizer; validation only
          -> results/tuning/{d}/{m}/cfg{k}.json
  smoke   default config, short training, everything exercised -> results_smoke/{device}/
  pilot   default config, full training + cost, validation only -> results_pilot/

Run modes (``--run-mode``; hardware concurrency only, never the protocol)
  quality  run alongside other jobs: predictive metrics + predictions only, no latency,
           ``timing_reportable: false`` -> results/raw/ (tuning: results/tuning_quality/)
  cost     exclusive and sequential: every cost metric, ``timing_reportable: true``
           -> results/cost/raw/ (tuning: results/tuning/)
Both modes use the same per-job thread counts and device, so predictions are identical
(up to cuDNN RNN nondeterminism).

Example::

    python -m ksent.run_one --dataset nsmc --method tfidf_lr --tokenizer kiwi --size 2000 --seed 13
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ksent.config import (SOURCE_DIR, family, frozen_config, global_cfg, main_tokenizer, max_seq_len, path,
                          processed_dir, resolve_config, resolve_size, search_space)
from ksent.data.normalize import normalize
from ksent.data.split import load_split
from ksent.data.subsets import load_subset
from ksent.eval.cost import RSSSampler, Timer, file_size_mb, measure_latency, ru_maxrss_mb
from ksent.eval.metrics import classification_metrics
from ksent.features.tokenize import cached_tokens, tfidf_uses_text, tokenize_many
from ksent.hardware import (AcceleratorMemoryTracker, compact_info, configured_device, record_mps_fallbacks,
                            resolve_device, resolve_mamba_backend, seed_everything, set_determinism, set_threads)
from ksent.utils import git_commit, now_iso, read_json, write_json


# ---------------------------------------------------------------------------- paths
def result_path(out_root: Path, mode: str, dataset: str, method: str, tokenizer: str, size: int, seed: int,
                config_id: int | None) -> Path:
    """Canonical JSON location of a run."""
    if mode == "tuning":
        return out_root / "tuning" / dataset / method / f"cfg{config_id}.json"
    return out_root / "raw" / dataset / method / tokenizer / f"n{size}_s{seed}.json"


def default_out_root(mode: str, device: str | None = None, run_mode: str = "cost") -> Path:
    """Result root of a mode: final quality runs in ``results/``, final cost runs in
    ``results/cost/``; tuning cost runs in ``results/`` (-> ``results/tuning/``), tuning
    quality runs in ``results/tuning_quality/``; separate roots for smoke/pilot."""
    if mode == "final":
        return path("results") / "cost" if run_mode == "cost" else path("results")
    if mode == "tuning":
        return path("results") if run_mode == "cost" else path("results") / "tuning_quality"
    if mode == "smoke":
        return SOURCE_DIR / "results_smoke" / (device or "auto").replace(":", "")
    return SOURCE_DIR / "results_pilot"


def is_done(p: Path, expect: dict[str, Any] | None = None) -> bool:
    """A run is complete when its JSON exists, has ``status == "ok"`` and (optionally)
    matches the expected identity fields."""
    if not p.exists():
        return False
    try:
        r = read_json(p)
    except Exception:
        return False
    if r.get("status") != "ok":
        return False
    return all(r.get(k) == v for k, v in (expect or {}).items())


# ---------------------------------------------------------------------------- helpers
def _latency(pipe, raw_texts: list[str], inputs: list, n_warmup: int, device: str) -> dict:
    """Classifier-only (prepared input) and end-to-end (raw string -> label) batch-1 latency."""
    prepared = [pipe.prepare([x]) for x in inputs]

    def e2e(raw: str) -> int:
        return pipe.predict_one_prepared(pipe.prepare([pipe.text_to_input(raw)]))

    return {"classifier_only": measure_latency(pipe.predict_one_prepared, prepared, n_warmup, device),
            "end_to_end": measure_latency(e2e, raw_texts, n_warmup, device)}


def _model_inputs(fam: str, tokenizer: str, dataset: str, frame: pd.DataFrame) -> list:
    """Cached classifier inputs for a split: token lists, or normalized text for char TF-IDF."""
    if fam == "tfidf" and tfidf_uses_text(tokenizer):
        return frame["text"].tolist()
    return cached_tokens(dataset, frame["review_id"], tokenizer)


def _oov(vocab, docs: list[list[str]], max_len: int) -> dict[str, float]:
    oov = [[t not in vocab.stoi for t in d] for d in docs]
    n_tok = sum(len(o) for o in oov)
    return {"oov_token_rate": sum(map(sum, oov)) / max(n_tok, 1),
            "oov_review_rate": float(np.mean([any(o) for o in oov])),
            "truncation_rate": float(np.mean([len(d) > max_len for d in docs]))}


# ---------------------------------------------------------------------------- core
def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one run and return the result record (also written to disk)."""
    g = global_cfg()
    n_threads = set_threads()
    fam0 = family(args.method)
    # neural jobs train with `neural_threads` CPU threads (in both run modes) so several can share a
    # GPU; every CPU latency measurement uses `n_threads`, the same count as the classical fits
    train_threads = int(g.get("neural_threads") or n_threads) if fam0 == "neural" else n_threads
    sampler = RSSSampler().start()
    t_start = time.perf_counter()
    method, dataset, tok, size, seed, mode = args.method, args.dataset, args.tokenizer, args.size, args.seed, args.mode
    fam = family(method)

    if args.config_id is not None:
        params, config_source = resolve_config(method, args.config_id), f"search_space:cfg{args.config_id}"
    elif mode in ("smoke", "pilot"):
        cid = int(search_space(method)["default_index"])
        params, config_source = resolve_config(method, cid), f"search_space:default(cfg{cid})"
    else:
        params, config_source = frozen_config(dataset, method), "frozen"
    tuning = mode == "tuning"
    evaluate_test = mode in ("final", "smoke")
    measure_cost = mode in ("final", "smoke", "pilot") and not args.no_latency

    # ---- device
    if fam == "tfidf":  # scikit-learn / XGBoost (device="cpu") have no .to(device)
        device, device_reason = "cpu", "classical methods always run on CPU"
    else:
        device, device_reason = resolve_device(args.device or configured_device(), allow_fallback=args.allow_fallback)
    print(f"device: {device} ({device_reason})", flush=True)

    rec: dict[str, Any] = {
        "status": "running", "mode": mode, "dataset": dataset, "method": method, "family": fam, "tokenizer": tok,
        "size": size, "seed": seed, "config_id": params.get("config_id"), "config_source": config_source,
        "config": params, "git_commit": git_commit(), "timestamp": now_iso(), "hostname": os.uname().nodename,
        "n_threads": n_threads, "train_device": device, "device_reason": device_reason,
        "precision": "amp_fp16" if args.amp else "fp32", "timing_reportable": not args.not_reportable,
        "run_mode": args.run_mode, "train_threads": train_threads,
        "hardware": compact_info(), "python": sys.version.split()[0],
    }

    # ---- data: train subset preprocessed live from raw text (timed); val/test from the cache
    seed_everything(seed, "cpu")
    train = load_subset(dataset, seed, size)
    val, test = load_split(dataset, "val"), load_split(dataset, "test")
    y_tr, y_val, y_te = (d["label"].to_numpy() for d in (train, val, test))
    char_tfidf = fam == "tfidf" and tfidf_uses_text(tok)
    tokenize_many(["워밍업 문장입니다"], tok)  # load the tokenizer model outside the timed region
    with Timer() as t_tok:
        norm = [normalize(t) for t in train["text_raw"]]
        train_inputs = norm if char_tfidf else tokenize_many(norm, tok)
    val_inputs = _model_inputs(fam, tok, dataset, val)
    test_inputs = _model_inputs(fam, tok, dataset, test)
    rec["data"] = {"n_train": len(train), "n_val": len(val), "n_test": len(test),
                   "train_pct_positive": float(y_tr.mean() * 100)}

    # ---- fit
    rss_before_fit = sampler.current_mb()
    impl: dict[str, Any] = {}
    if fam == "tfidf":
        from ksent.train.fit_classical import fit_classical

        pipe, info = fit_classical(method, params, train_inputs, y_tr, seed, n_threads, tok,
                                   tuning=tuning, val_inputs=val_inputs, y_val=y_val)
    else:
        from ksent.train.fit_neural import fit_neural

        set_threads(train_threads)  # torch is loaded now -> apply the thread limit to it
        rec.update(set_determinism(device))
        msl = max_seq_len(tok)
        backend = resolve_mamba_backend(args.mamba_backend, device) if method == "small_mamba" else None
        pipe, info = fit_neural(method, params, train_inputs, y_tr, val_inputs, y_val, seed, device, msl, tok,
                                mamba_backend=backend or "torch_ref", max_epochs=args.max_epochs, amp=args.amp)
        impl = {"max_seq_len": msl, "width": info["width"]}
        rec["model_device"] = info["model_device"]
        rec["mps_fallbacks"] = info.get("mps_fallbacks", [])
        rec["mps_fallback_used"] = bool(rec["mps_fallbacks"])
    rec["mamba_backend"] = info.get("mamba_backend")
    rec["implementation"] = impl
    rec["fit"] = {k: v for k, v in info.items() if k not in ("history", "mps_fallbacks")}
    rec["fit"]["tokenize_time_s"] = t_tok.seconds
    rec["history"] = info.get("history")
    rec["cost"] = {
        "fit_time_classifier_s": info["classifier_time_s"],
        "fit_time_end_to_end_s": t_tok.seconds + info["featurize_time_s"] + info["classifier_time_s"],
        "tokenize_time_s": t_tok.seconds, "featurize_time_s": info["featurize_time_s"],
        "peak_accelerator_mem_train_mb": info.get("peak_accelerator_mem_train_mb"),
    }

    # ---- evaluation (validation always; test only in final/smoke mode)
    vp, _ = pipe.predict_inputs(val_inputs)
    rec["val"] = classification_metrics(y_val, vp)
    if fam == "neural":
        rec["preprocessing"] = _oov(pipe.vocab, test_inputs if evaluate_test else val_inputs, pipe.max_len)
        rec["preprocessing"]["evaluated_on"] = "test" if evaluate_test else "val"
    if evaluate_test:
        tp, tprob = pipe.predict_inputs(test_inputs)
        rec["test"] = classification_metrics(y_te, tp)
        pred_path = Path(str(args.result_path).replace(f"{os.sep}raw{os.sep}", f"{os.sep}predictions{os.sep}")).with_suffix(".csv")
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"review_id": test["review_id"], "label": y_te, "pred": tp,
                      "prob_pos": tprob if tprob is not None else np.nan}).to_csv(pred_path, index=False)
        rec["predictions_path"] = str(pred_path)

    # ---- model size, serialised size, latency
    if fam == "tfidf":
        from ksent.models.classical import model_size_stats

        rec["model_size"] = {**model_size_stats(method, pipe.clf), "n_features": info["n_features"]}
    else:
        rec["model_size"] = {**info["params"], "vocab_size": info["vocab_size"]}

    # artifacts: quality runs at n=8,000, main tokenizer, demo seed only (disk budget);
    # serialized size is a cost metric and is recorded by every cost-measuring run
    art_cfg = g["artifacts"]
    if (mode == "final" and args.run_mode == "quality" and tok == main_tokenizer()
            and size == int(art_cfg["save_models_size"]) and seed == int(art_cfg["demo_seed"])):
        art = pipe.save(path("results") / "models" / dataset / method / f"n{size}_s{seed}")
        rec["artifact_path"] = str(art)
        rec["cost"]["serialized_mb"] = file_size_mb(art)

    if measure_cost:
        tmp = Path(tempfile.mkdtemp(prefix="ksent_"))
        rec["cost"]["serialized_mb"] = file_size_mb(pipe.save(tmp / "model"))
        shutil.rmtree(tmp, ignore_errors=True)

        lat_cfg = g["latency"]
        lat_ids = read_json(processed_dir(dataset) / "latency_ids.json")["review_ids"]
        lat_frame = test.set_index("review_id").loc[lat_ids].reset_index()
        lat_raw = lat_frame["text_raw"].tolist()
        lat_inputs = _model_inputs(fam, tok, dataset, lat_frame)
        latency: dict[str, Any] = {}
        set_threads(n_threads)
        if lat_cfg["cpu"]:
            cpu_pipe = pipe if fam == "tfidf" else pipe.to("cpu")
            latency["cpu"] = _latency(cpu_pipe, lat_raw, lat_inputs, int(lat_cfg["n_warmup"]), "cpu")
            if fam == "neural":
                latency["cpu"]["mamba_backend"] = cpu_pipe.mamba_backend
        if fam == "neural" and lat_cfg["accelerator"] and device != "cpu":
            fallbacks: list[str] = []
            mem = AcceleratorMemoryTracker(device)
            with record_mps_fallbacks(device, fallbacks):
                latency["accelerator"] = _latency(pipe, lat_raw, lat_inputs, int(lat_cfg["n_warmup"]), device)
            mem.sample()
            latency["accelerator"]["mamba_backend"] = pipe.mamba_backend
            rec["cost"]["peak_accelerator_mem_infer_mb"] = mem.peak_mb()
            if fallbacks:
                rec.setdefault("mps_fallbacks", []).extend(fallbacks)
                rec["mps_fallback_used"] = True
        rec["latency"] = latency
        rec["latency_devices"] = [v["classifier_only"]["device"] for v in latency.values()]

    rec["cost"]["peak_rss_sampled_mb"] = sampler.stop()
    rec["cost"]["peak_rss_mb"] = max(ru_maxrss_mb(), rec["cost"]["peak_rss_sampled_mb"])
    # memory attributable to fitting/evaluating (process peak minus RSS after imports,
    # tokenizer and data loading)
    rec["cost"]["rss_before_fit_mb"] = rss_before_fit
    rec["cost"]["peak_rss_delta_mb"] = rec["cost"]["peak_rss_sampled_mb"] - rss_before_fit
    rec["wall_time_s"] = time.perf_counter() - t_start
    rec["status"] = "ok"
    return rec


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--tokenizer", default=None, help="kiwi (main) | whitespace | char")
    ap.add_argument("--size", required=True, help="training size or 'full'")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config-id", type=int, default=None)
    ap.add_argument("--mode", choices=["final", "tuning", "smoke", "pilot"], default="final")
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--device", default=None, help="auto | cuda | cuda:N | mps | cpu (default: config/KSENT_DEVICE)")
    ap.add_argument("--allow-fallback", action="store_true", help="fall back if the device is missing (smoke only)")
    ap.add_argument("--mamba-backend", default=None, help="auto | mamba_ssm | torch_ref (default: config)")
    ap.add_argument("--amp", action="store_true", help="fp16 autocast on CUDA (experiments only, never reported)")
    ap.add_argument("--max-epochs", type=int, default=None, help="override (smoke test only)")
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--not-reportable", action="store_true", help="mark timings as not reportable (parallel runs)")
    ap.add_argument("--force", action="store_true", help="re-run even if a valid result exists")
    ap.add_argument("--run-mode", choices=["quality", "cost"], default="cost",
                    help="quality: parallel-safe, no latency, timings not reportable; cost: exclusive (default)")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    args.tokenizer = args.tokenizer or main_tokenizer()
    args.size = resolve_size(args.dataset, args.size)
    if args.mode == "tuning":
        if args.config_id is None:
            ap.error("--mode tuning requires --config-id")
        if args.tokenizer != main_tokenizer():
            ap.error("tuning always uses the main tokenizer")
    if args.mode == "final" and (args.allow_fallback or args.amp or args.max_epochs):
        ap.error("--allow-fallback/--amp/--max-epochs are not allowed for final (reported) runs")
    if args.run_mode == "quality":
        args.not_reportable = True
        args.no_latency = True
    elif args.not_reportable and args.mode in ("final", "tuning"):
        ap.error("--not-reportable with --run-mode cost: use --run-mode quality for parallel runs")
    out_root = args.out_root or default_out_root(args.mode, args.device, args.run_mode)
    args.result_path = result_path(out_root, args.mode, args.dataset, args.method, args.tokenizer, args.size,
                                   args.seed, args.config_id)
    if is_done(args.result_path) and not args.force:
        print(f"skip (exists): {args.result_path}")
        return 0
    failed_path = args.result_path.with_suffix(".failed.json")
    try:
        rec = run(args)
    except Exception as e:
        write_json({"status": "failed", "error": repr(e), "traceback": traceback.format_exc(),
                    "argv": sys.argv, "timestamp": now_iso()}, failed_path)
        traceback.print_exc()
        return 1
    write_json(rec, args.result_path)
    if failed_path.exists():
        failed_path.unlink()
    test_f1 = rec.get("test", {}).get("macro_f1")
    print(f"ok {args.dataset} {args.method} {args.tokenizer} n={args.size} seed={args.seed} [{rec['train_device']}] "
          f"val_f1={rec['val']['macro_f1']:.4f}" + (f" test_f1={test_f1:.4f}" if test_f1 is not None else "")
          + f" fit={rec['cost']['fit_time_end_to_end_s']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
