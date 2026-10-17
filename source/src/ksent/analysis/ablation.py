"""Tokenizer ablation: does the crossover conclusion depend on the tokenizer?

Methods: per dataset, the best TF-IDF and the best neural method by mean *validation*
macro-F1 at n=8,000 in the main (Kiwi) sweep. Tokenizers: whitespace, kiwi, char.
Sizes: 500 and 8,000; all 5 seeds; frozen (Kiwi-tuned) hyperparameters reused.

Outputs (results/aggregated/):
  ablation_methods.json  the selected methods
  ablation_cells.csv     test macro-F1 (mean ± std) + vocab / OOV per method x tokenizer x size
  ablation_gap.csv       neural - TF-IDF gap with each family's best tokenizer (chosen on
                         validation) vs the main-tokenizer gap, and whether the conclusion holds
Usage: ``python -m ksent.analysis.ablation``
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ksent.analysis.common import AGG_DIR, read_csv
from ksent.config import NEURAL_METHODS, TFIDF_METHODS, family, global_cfg, main_tokenizer
from ksent.utils import write_json


def ablation_methods(runs: pd.DataFrame | None = None, require_complete: bool = True) -> dict[str, list[str]]:
    """``{dataset: [best_tfidf, best_neural]}`` from the main sweep at the selection size."""
    from ksent.analysis.aggregate import load_runs

    g = global_cfg()
    runs = load_runs() if runs is None else runs
    n_sel = int(g["ablation"]["selection_size"])
    if runs.empty:
        raise SystemExit("ablation: no main-sweep results yet - run scripts/05_run_all.sh first")
    sub = runs[(runs.tokenizer == main_tokenizer()) & (runs["size"] == n_sel)]
    out: dict[str, list[str]] = {}
    for d in g["datasets"]:
        cell = sub[sub.dataset == d]
        if require_complete:
            counts = cell.groupby("method").seed.nunique()
            missing = [m for m in g["methods"] if counts.get(m, 0) < len(g["seeds"])]
            if missing:
                raise SystemExit(f"ablation: main sweep incomplete for {d} n={n_sel}: {missing}")
        means = cell.groupby("method").val_macro_f1.mean()
        picks = []
        for fam_methods in (TFIDF_METHODS, NEURAL_METHODS):
            m = means[means.index.isin(fam_methods)]
            if len(m):
                picks.append(str(m.idxmax()))
        out[d] = picks
    return out


def _paired(runs: pd.DataFrame, d: str, n: int, a: tuple[str, str], b: tuple[str, str]) -> pd.Series:
    """Per-seed (a - b) test macro-F1 in points; a, b = (method, tokenizer)."""
    sub = runs[(runs.dataset == d) & (runs["size"] == n)]
    x = sub[(sub.method == a[0]) & (sub.tokenizer == a[1])].set_index("seed").test_macro_f1
    y = sub[(sub.method == b[0]) & (sub.tokenizer == b[1])].set_index("seed").test_macro_f1
    return ((x - y) * 100).dropna()


def analyse(runs: pd.DataFrame, chosen: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = global_cfg()
    ab = g["ablation"]
    margin = float(g["analysis"]["crossover_margin"])
    cells, gaps = [], []
    for d, ms in chosen.items():
        for m in ms:
            for t in ab["tokenizers"]:
                for n in ab["sizes"]:
                    c = runs[(runs.dataset == d) & (runs.method == m) & (runs.tokenizer == t) & (runs["size"] == n)]
                    if c.empty:
                        continue
                    row = {"dataset": d, "method": m, "family": family(m), "tokenizer": t, "size": n,
                           "n_seeds": c.seed.nunique()}
                    for col in ("test_macro_f1", "val_macro_f1", "vocab_size", "n_features", "oov_token_rate",
                                "oov_review_rate", "truncation_rate", "fit_e2e_s", "lat_cpu_e2e_ms"):
                        if col in c:
                            row[f"{col}_mean"] = c[col].mean()
                            row[f"{col}_std"] = c[col].std(ddof=1) if len(c) > 1 else np.nan
                    cells.append(row)
    cells_df = pd.DataFrame(cells)
    for d, ms in chosen.items():
        if len(ms) < 2 or cells_df.empty:
            continue
        t_m, n_m = ms
        for n in ab["sizes"]:
            sub = cells_df[(cells_df.dataset == d) & (cells_df["size"] == n)]
            best = {}
            for m in (t_m, n_m):
                s = sub[sub.method == m].dropna(subset=["val_macro_f1_mean"])
                best[m] = str(s.sort_values("val_macro_f1_mean", ascending=False).iloc[0].tokenizer) if len(s) else None
            row = {"dataset": d, "size": n, "tfidf_method": t_m, "neural_method": n_m,
                   "best_tokenizer_tfidf": best[t_m], "best_tokenizer_neural": best[n_m]}
            gm = _paired(runs, d, n, (n_m, main_tokenizer()), (t_m, main_tokenizer()))
            row.update(gap_main_mean=gm.mean() if len(gm) else np.nan,
                       gap_main_std=gm.std(ddof=1) if len(gm) > 1 else np.nan)
            if best[t_m] and best[n_m]:
                gb = _paired(runs, d, n, (n_m, best[n_m]), (t_m, best[t_m]))
                row.update(gap_best_mean=gb.mean() if len(gb) else np.nan,
                           gap_best_std=gb.std(ddof=1) if len(gb) > 1 else np.nan, n_pairs=len(gb))
            if pd.notna(row.get("gap_main_mean")) and pd.notna(row.get("gap_best_mean")):
                row["conclusion_holds"] = bool((row["gap_main_mean"] >= margin) == (row["gap_best_mean"] >= margin))
            gaps.append(row)
    return cells_df, pd.DataFrame(gaps)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    runs = read_csv("runs.csv")
    outs = ("ablation_cells.csv", "ablation_gap.csv", "ablation_methods.json")
    if runs is None or (runs.tokenizer != main_tokenizer()).sum() == 0:
        print("no ablation results yet - skipping (run scripts/06_run_ablation.sh)")
        for f in outs:
            (AGG_DIR / f).unlink(missing_ok=True)
        return
    chosen = ablation_methods(runs, require_complete=False)
    write_json(chosen, AGG_DIR / "ablation_methods.json")
    cells, gaps = analyse(runs, chosen)
    cells.to_csv(AGG_DIR / "ablation_cells.csv", index=False)
    gaps.to_csv(AGG_DIR / "ablation_gap.csv", index=False)
    for _, r in gaps.iterrows():
        print(f"[{r.dataset:8s}] n={r['size']}: best tok TF-IDF={r.best_tokenizer_tfidf} neural={r.best_tokenizer_neural} "
              f"gap main={r.gap_main_mean:.2f} best-tok={r.get('gap_best_mean', np.nan):.2f} "
              f"holds={r.get('conclusion_holds')}")


if __name__ == "__main__":
    main()
