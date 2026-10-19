"""Compare cost-mode with quality-mode test predictions of the same runs.

For every run present in both ``results/cost/predictions/`` and ``results/predictions/``
(same dataset, method, tokenizer, size, seed), compute the fraction of test reviews whose
predicted label differs. Classical methods must match exactly; neural methods may differ
slightly (cuDNN RNN nondeterminism). Any method whose disagreement exceeds 0.5% of test
predictions in any run is a stop condition (exit code 1).

Output: ``results/aggregated/determinism.csv`` (per run) and ``determinism_by_method.csv``.
Usage: ``python -m ksent.analysis.determinism``
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from ksent.analysis.common import AGG_DIR
from ksent.config import family, path

THRESHOLD = 0.005


def main() -> int:
    q_root, c_root = path("results") / "predictions", path("results") / "cost" / "predictions"
    rows = []
    for cp in sorted(glob.glob(str(c_root / "*" / "*" / "*" / "*.csv"))):
        rel = Path(cp).relative_to(c_root)
        qp = q_root / rel
        d, m, tok, name = rel.parts
        if not qp.exists():
            rows.append({"dataset": d, "method": m, "tokenizer": tok, "run": name, "missing_quality": True})
            continue
        a, b = pd.read_csv(qp), pd.read_csv(cp)
        j = a.merge(b, on="review_id", suffixes=("_q", "_c"))
        if len(j) != len(a) or len(j) != len(b):
            raise SystemExit(f"{rel}: review ids differ between quality and cost predictions")
        diff = float((j.pred_q != j.pred_c).mean())
        prob = (j.prob_pos_q - j.prob_pos_c).abs().max() if "prob_pos_q" in j else float("nan")
        rows.append({"dataset": d, "method": m, "family": family(m), "tokenizer": tok, "run": name.removesuffix(".csv"),
                     "n": len(j), "n_disagree": int((j.pred_q != j.pred_c).sum()), "disagree_frac": diff,
                     "max_abs_prob_diff": prob, "missing_quality": False})
    df = pd.DataFrame(rows)
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    if df.empty:
        print("no cost-mode predictions found")
        return 1
    df.to_csv(AGG_DIR / "determinism.csv", index=False)
    ok = df[~df.missing_quality]
    by = ok.groupby(["method", "family"]).agg(runs=("run", "count"), identical_runs=("n_disagree", lambda s: int((s == 0).sum())),
                                              mean_disagree_pct=("disagree_frac", lambda s: 100 * s.mean()),
                                              max_disagree_pct=("disagree_frac", lambda s: 100 * s.max()),
                                              max_abs_prob_diff=("max_abs_prob_diff", "max")).reset_index()
    by.to_csv(AGG_DIR / "determinism_by_method.csv", index=False)
    pd.set_option("display.width", 200)
    print(by.to_string(index=False))
    if df.missing_quality.any():
        print(f"WARNING: {int(df.missing_quality.sum())} cost runs have no quality counterpart")
    bad_classical = ok[(ok.family == "tfidf") & (ok.n_disagree > 0)]
    bad = ok[ok.disagree_frac > THRESHOLD]
    if len(bad_classical):
        print(f"WARNING: classical methods differ between modes in {len(bad_classical)} runs:")
        print(bad_classical.to_string(index=False))
    if len(bad):
        print(f"STOP: {len(bad)} runs disagree on > {THRESHOLD:.1%} of test predictions:")
        print(bad.to_string(index=False))
        return 1
    print(f"determinism OK: no run disagrees on more than {THRESHOLD:.1%} of test predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
