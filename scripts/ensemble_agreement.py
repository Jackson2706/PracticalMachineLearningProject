"""How often do TF-IDF and the pooled Transformer disagree, and who is right when they do?

Reads the per-review probabilities saved by scripts/ensemble_followup.py
(source/results/followup_ensemble/cells/*/probabilities.npz and metrics.csv) and writes
source/results/followup_ensemble/agreement_summary.csv. No training, no GPU.

Usage, from the repository root:
    python scripts/ensemble_agreement.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "source" / "results" / "followup_ensemble"
NEURAL = "pooled_ce"  # the pooled Transformer used on the slides

rows = []
for cell in sorted((ROOT / "cells").glob("*_*_*")):
    if not (cell / "SUCCESS").exists():
        continue
    target, size, seed = cell.name.rsplit("_", 2)
    data = np.load(cell / "probabilities.npz", allow_pickle=True)
    metrics = pd.read_csv(cell / "metrics.csv")
    w = float(metrics.loc[metrics.method == f"ens_valw_{NEURAL}", "weight_tfidf"].iloc[0])
    y = data["test_label"].astype(int)
    p_t, p_n = data["test_p_tfidf"], data[f"test_p_{NEURAL}"]
    pred_t, pred_n = (p_t >= 0.5).astype(int), (p_n >= 0.5).astype(int)
    pred_e = ((w * p_t + (1 - w) * p_n) >= 0.5).astype(int)
    disagree = pred_t != pred_n
    rows.append({
        "target_domain": target, "target_train_size": int(size), "seed": int(seed), "w_tfidf": w,
        "disagreement_rate": disagree.mean(),
        "both_wrong_rate": ((pred_t != y) & (pred_n != y)).mean(),
        "tfidf_correct_when_disagree": (pred_t[disagree] == y[disagree]).mean() if disagree.any() else np.nan,
        "neural_correct_when_disagree": (pred_n[disagree] == y[disagree]).mean() if disagree.any() else np.nan,
        "ensemble_correct_when_disagree": (pred_e[disagree] == y[disagree]).mean() if disagree.any() else np.nan,
        "probability_correlation": np.corrcoef(p_t, p_n)[0, 1],
    })

if not rows:
    raise SystemExit(f"no finished cells under {ROOT / 'cells'}")
by_seed = pd.DataFrame(rows)
summary = by_seed.drop(columns="seed").groupby(["target_domain", "target_train_size"]).mean().round(3).reset_index()
by_seed.to_csv(ROOT / "agreement_by_seed.csv", index=False)
summary.to_csv(ROOT / "agreement_summary.csv", index=False)
print(summary.to_string(index=False))
