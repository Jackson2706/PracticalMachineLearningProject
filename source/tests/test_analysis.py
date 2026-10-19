"""Analysis chain on synthetic results: aggregation, paired crossover, tables, numbers."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
G = {"datasets": ["nsmc", "shopping", "steam"], "sizes": [500, 1000, 2000, 4000, 8000],
     "seeds": [13, 42, 87, 2024, 31337]}
METHODS = ["tfidf_lr", "tfidf_svm", "tfidf_rf", "tfidf_xgb", "emb_mlp", "emb_lstm", "emb_gru",
           "small_transformer", "small_mamba"]


def _fake_run(d: str, m: str, n: int, s: int, fp: str = "abc123", tok: str = "kiwi", cost_scale: float = 1.0,
              reportable: bool = True) -> dict:
    """TF-IDF LR is flat at 0.80; GRU starts lower and gains 2.5 pts per doubling.
    -> GRU - LR gap: 500:-4, 1000:-1.5, 2000:+1, ... => crossover at 2,000."""
    k = [500, 1000, 2000, 4000, 8000].index(n)
    jitter = (s % 7) / 1e4
    if m == "tfidf_lr":
        f1 = 0.80 + jitter
    elif m == "emb_gru":
        f1 = 0.76 + 0.025 * k + jitter
    else:
        f1 = 0.70 + jitter
    neural = m.startswith(("emb", "small"))
    return {"status": "ok", "dataset": d, "method": m, "family": "neural" if neural else "tfidf", "size": n,
            "seed": s, "tokenizer": tok, "config_id": 0, "timing_reportable": reportable, "train_device": "cpu",
            "precision": "fp32", "n_threads": 8, "hardware": {"fingerprint": fp, "cpu_model": "x", "gpu_name": None},
            "mamba_backend": "torch_ref" if m == "small_mamba" else None,
            "val": {"macro_f1": f1, "accuracy": f1, "macro_precision": f1, "macro_recall": f1,
                    "confusion": [[800, 200], [200, 800]]},
            "test": {"macro_f1": f1, "accuracy": f1, "macro_precision": f1, "macro_recall": f1,
                     "confusion": [[810, 190], [210, 790]]},
            "cost": {"fit_time_end_to_end_s": (10.0 if neural else 1.0) * cost_scale,
                     "fit_time_classifier_s": 9.0 if neural else 0.5,
                     "tokenize_time_s": 0.1, "featurize_time_s": 0.1, "peak_rss_mb": 900.0 if neural else 500.0,
                     "serialized_mb": 2.0},
            "latency": {"cpu": {"end_to_end": {"median_ms": (2.0 if neural else 0.5) * cost_scale, "p95_ms": 3.0},
                                "classifier_only": {"median_ms": 1.0, "p95_ms": 1.5}}},
            "model_size": {}, "fit": {"best_epoch": 3, "epochs_run": 8}, "wall_time_s": 20.0}


def _run(mod: str, env: dict) -> str:
    out = subprocess.run([sys.executable, "-m", mod], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout


def _env(tmp: Path) -> dict:
    return {**os.environ, "PYTHONPATH": str(SRC), "KSENT_RESULTS_DIR": str(tmp / "results"),
            "KSENT_REPORT_DIR": str(tmp / "report")}


def test_tables_and_numbers_without_results(tmp_path: Path) -> None:
    env = _env(tmp_path)
    for mod in ("ksent.analysis.aggregate", "ksent.analysis.crossover", "ksent.analysis.make_tables",
                "ksent.analysis.make_numbers"):
        _run(mod, env)
    t3 = (tmp_path / "report" / "tables" / "table3.tex").read_text()
    assert "--" in t3 and r"\toprule" in t3
    nums = (tmp_path / "report" / "generated" / "numbers.tex").read_text()
    assert r"\newcommand{\crossoverNSMC}{\TBD{crossoverNSMC}}" in nums


def test_full_chain_on_synthetic_results(tmp_path: Path) -> None:
    """Quality runs carry 100x inflated (parallel) costs; only the exclusive cost runs
    (n=8,000 and the crossover size 2,000) may reach the tables."""
    raw, cost = tmp_path / "results" / "raw", tmp_path / "results" / "cost" / "raw"
    for d in G["datasets"]:
        for m in METHODS:
            for n in G["sizes"]:
                for s in G["seeds"]:
                    p = raw / d / m / "kiwi" / f"n{n}_s{s}.json"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(json.dumps(_fake_run(d, m, n, s, cost_scale=100.0, reportable=False)))
                    if n in (2000, 8000):
                        c = cost / d / m / "kiwi" / f"n{n}_s{s}.json"
                        c.parent.mkdir(parents=True, exist_ok=True)
                        c.write_text(json.dumps(_fake_run(d, m, n, s)))
    env = _env(tmp_path)
    _run("ksent.analysis.aggregate", env)
    _run("ksent.analysis.crossover", env)
    agg = tmp_path / "results" / "aggregated"
    summary = pd.read_csv(agg / "summary.csv")
    assert len(summary) == 3 * 9 * 5 and (summary.n_seeds == 5).all()
    lr = summary[summary.method == "tfidf_lr"].set_index(["dataset", "size"])
    assert (lr.xs(8000, level="size").fit_e2e_s_mean == 1.0).all()            # from cost runs
    assert (lr.xs(8000, level="size").lat_cpu_e2e_ms_mean == 0.5).all()
    assert lr.xs(500, level="size").fit_e2e_s_mean.isna().all()               # no cost run -> no cost value
    assert lr.xs(8000, level="size").all_timings_reportable.all()
    assert not lr.xs(500, level="size").all_timings_reportable.any()
    ct = pd.read_csv(agg / "crossover.csv")
    v = ct[ct.selection == "val"].set_index("dataset")
    assert (v.threshold == 2000).all()
    assert (v.best_neural == "emb_gru").all() and (v.best_tfidf == "tfidf_lr").all()
    assert abs(v.loc["nsmc", "gap_mean"] - 1.0) < 1e-6   # 0.76 + 2*0.025 - 0.80 = +1 point
    assert (v.neural_pareto_efficient == True).all()     # noqa: E712  (higher F1, so not dominated)
    _run("ksent.analysis.make_tables", env)
    _run("ksent.analysis.make_numbers", env)
    t5 = (tmp_path / "report" / "tables" / "table5.tex").read_text()
    assert "2{,}000" in t5 and "Emb + GRU" in t5
    nums = (tmp_path / "report" / "generated" / "numbers.tex").read_text()
    assert r"\newcommand{\crossoverNSMC}{2{,}000}" in nums
    out = subprocess.run([sys.executable, "-c", "from ksent.analysis.crossover import crossover_cost_jobs as f; "
                          "j = f(); print(len(j), sorted({(x.method, x.size) for x in j}))"],
                         env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("30 [('emb_gru', 2000), ('tfidf_lr', 2000)]")


def test_mixed_hardware_refused(tmp_path: Path) -> None:
    raw = tmp_path / "results" / "raw" / "nsmc" / "tfidf_lr" / "kiwi"
    raw.mkdir(parents=True)
    (raw / "n500_s13.json").write_text(json.dumps(_fake_run("nsmc", "tfidf_lr", 500, 13, fp="machineA")))
    (raw / "n500_s42.json").write_text(json.dumps(_fake_run("nsmc", "tfidf_lr", 500, 42, fp="machineB")))
    env = _env(tmp_path)
    out = subprocess.run([sys.executable, "-m", "ksent.analysis.aggregate"], env=env, capture_output=True, text=True)
    assert out.returncode != 0 and "fingerprint" in (out.stderr + out.stdout)
    out = subprocess.run([sys.executable, "-m", "ksent.analysis.aggregate", "--allow-mixed-hardware"], env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0 and "WARNING" in out.stdout


def test_ablation_chain_on_synthetic_results(tmp_path: Path) -> None:
    """Main sweep: GRU beats LR from 2,000 on. Ablation: char TF-IDF gains +3 points,
    which shrinks (but does not remove) the neural advantage at 8,000."""
    raw = tmp_path / "results" / "raw"
    for d in G["datasets"]:
        for m in METHODS:
            for n in G["sizes"]:
                for s in G["seeds"]:
                    p = raw / d / m / "kiwi" / f"n{n}_s{s}.json"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(json.dumps(_fake_run(d, m, n, s)))
        for m in ("tfidf_lr", "emb_gru"):
            for tok in ("whitespace", "char"):
                for n in (500, 8000):
                    for s in G["seeds"]:
                        r = _fake_run(d, m, n, s, tok=tok)
                        bump = 0.03 if (m == "tfidf_lr" and tok == "char") else -0.01
                        for split in ("val", "test"):
                            r[split]["macro_f1"] += bump
                        p = raw / d / m / tok / f"n{n}_s{s}.json"
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_text(json.dumps(r))
    env = _env(tmp_path)
    for mod in ("ksent.analysis.aggregate", "ksent.analysis.crossover", "ksent.analysis.ablation",
                "ksent.analysis.make_tables", "ksent.analysis.make_numbers"):
        _run(mod, env)
    agg = tmp_path / "results" / "aggregated"
    gaps = pd.read_csv(agg / "ablation_gap.csv")
    big = gaps[(gaps.dataset == "nsmc") & (gaps["size"] == 8000)].iloc[0]
    assert big.tfidf_method == "tfidf_lr" and big.neural_method == "emb_gru"
    assert big.best_tokenizer_tfidf == "char" and big.best_tokenizer_neural == "kiwi"
    assert abs(big.gap_main_mean - 6.0) < 1e-6 and abs(big.gap_best_mean - 3.0) < 1e-6
    assert bool(big.conclusion_holds)  # still >= 1 point at 8,000
    small = gaps[(gaps.dataset == "nsmc") & (gaps["size"] == 500)].iloc[0]
    assert small.gap_best_mean < small.gap_main_mean
    # main crossover is unaffected by the ablation runs (kiwi only)
    ct = pd.read_csv(agg / "crossover.csv")
    assert (ct[ct.selection == "val"].threshold == 2000).all()
    t6 = (tmp_path / "report" / "tables" / "table6.tex").read_text()
    assert "Char" in t6 and "TF-IDF + LR" in t6
    nums = (tmp_path / "report" / "generated" / "numbers.tex").read_text()
    assert r"\newcommand{\bestTokenizerTfidfNSMC}{character}" in nums
