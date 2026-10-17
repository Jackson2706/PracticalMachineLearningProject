"""Crossover analysis between the best TF-IDF and the best neural pipeline.

For each dataset and training size:
  1. pick the best TF-IDF and the best neural method by mean *validation* macro-F1;
  2. gap = neural - TF-IDF test macro-F1 (x100), computed per seed (paired: same
     training subset), then mean ± std (ddof=1);
  3. threshold = smallest size whose mean gap >= ``analysis.crossover_margin``;
  4. sensitivity: the same with winners chosen by test macro-F1;
  5. Pareto check of the neural vs the TF-IDF winner at the threshold (or the
     largest size) on (test macro-F1 ↑, CPU end-to-end latency ↓, fit time ↓, peak RAM ↓);
  6. marginal value of labels: per-seed macro-F1 gain between adjacent sizes.

Outputs (results/aggregated/): crossover_by_size.csv, crossover.csv, pareto_all.csv,
marginal_gains.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ksent.analysis.common import AGG_DIR, read_csv
from ksent.config import NEURAL_METHODS, TFIDF_METHODS, global_cfg, main_tokenizer

# objective -> +1 maximise, -1 minimise
PARETO_OBJECTIVES: dict[str, int] = {"test_macro_f1": +1, "lat_cpu_e2e_ms": -1, "fit_e2e_s": -1, "peak_rss_mb": -1}


def _best(summary: pd.DataFrame, dataset: str, size: int, fam_methods: tuple[str, ...], by: str) -> str | None:
    cell = summary[(summary.dataset == dataset) & (summary["size"] == size) & summary.method.isin(fam_methods)]
    cell = cell.dropna(subset=[f"{by}_mean"])
    if cell.empty:
        return None
    return str(cell.sort_values(f"{by}_mean", ascending=False).iloc[0].method)


def paired_gap(runs: pd.DataFrame, dataset: str, size: int, neural: str, tfidf: str) -> pd.Series:
    """Per-seed neural - TF-IDF test macro-F1 in points (x100), seeds present in both."""
    sub = runs[(runs.dataset == dataset) & (runs["size"] == size)]
    a = sub[sub.method == neural].set_index("seed").test_macro_f1
    b = sub[sub.method == tfidf].set_index("seed").test_macro_f1
    return ((a - b) * 100).dropna()


def by_size(runs: pd.DataFrame, summary: pd.DataFrame, selection: str) -> pd.DataFrame:
    """Winners and paired gaps for every (dataset, size)."""
    by = "val_macro_f1" if selection == "val" else "test_macro_f1"
    rows = []
    for d in global_cfg()["datasets"]:
        for n in global_cfg()["sizes"]:
            t, nn = _best(summary, d, n, TFIDF_METHODS, by), _best(summary, d, n, NEURAL_METHODS, by)
            if t is None or nn is None:
                continue
            g = paired_gap(runs, d, n, nn, t)
            rows.append({"dataset": d, "size": n, "selection": selection, "best_tfidf": t, "best_neural": nn,
                         "gap_mean": g.mean() if len(g) else np.nan,
                         "gap_std": g.std(ddof=1) if len(g) > 1 else np.nan, "n_pairs": len(g)})
    return pd.DataFrame(rows)


def dominates(a: dict, b: dict) -> bool:
    """True if ``a`` is at least as good as ``b`` on every objective and better on one."""
    better = False
    for k, sign in PARETO_OBJECTIVES.items():
        va, vb = a.get(k), b.get(k)
        if va is None or vb is None or np.isnan(va) or np.isnan(vb):
            return False
        if sign * (va - vb) < 0:
            return False
        better |= sign * (va - vb) > 0
    return better


def _means(summary: pd.DataFrame, d: str, n: int, m: str) -> dict:
    row = summary[(summary.dataset == d) & (summary["size"] == n) & (summary.method == m)]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {k: float(r.get(f"{k}_mean", np.nan)) for k in PARETO_OBJECTIVES} | {
        "lat_cpu_clf_ms": float(r.get("lat_cpu_clf_ms_mean", np.nan)), "fit_clf_s": float(r.get("fit_clf_s_mean", np.nan))}


def crossover_table(bs: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, selection): threshold, gap, cost of both winners, Pareto verdict."""
    margin = float(global_cfg()["analysis"]["crossover_margin"])
    fallback = int(global_cfg()["analysis"]["pareto_size_fallback"])
    rows = []
    for (d, sel), grp in bs.groupby(["dataset", "selection"], sort=False):
        grp = grp.sort_values("size")
        hit = grp[grp.gap_mean >= margin]
        thr = int(hit["size"].iloc[0]) if len(hit) else None
        n_eval = thr if thr is not None else fallback
        at = grp[grp["size"] == n_eval]
        row = {"dataset": d, "selection": sel, "threshold": thr, "eval_size": n_eval,
               "sizes_available": len(grp), "complete": len(grp) == len(global_cfg()["sizes"])}
        if len(at):
            a = at.iloc[0]
            row.update(best_tfidf=a.best_tfidf, best_neural=a.best_neural, gap_mean=a.gap_mean, gap_std=a.gap_std,
                       n_pairs=a.n_pairs)
            t, nn = _means(summary, d, n_eval, a.best_tfidf), _means(summary, d, n_eval, a.best_neural)
            for k, v in t.items():
                row[f"tfidf_{k}"] = v
            for k, v in nn.items():
                row[f"neural_{k}"] = v
            row["neural_pareto_efficient"] = (not dominates(t, nn)) if t and nn else None
            row["neural_dominates_tfidf"] = dominates(nn, t) if t and nn else None
            ok = t and nn and t.get("lat_cpu_e2e_ms") and t.get("fit_e2e_s")
            row["latency_ratio"] = nn["lat_cpu_e2e_ms"] / t["lat_cpu_e2e_ms"] if ok else np.nan
            row["fit_ratio"] = nn["fit_e2e_s"] / t["fit_e2e_s"] if ok else np.nan
            row["ram_ratio"] = nn["peak_rss_mb"] / t["peak_rss_mb"] if ok else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pareto_all(summary: pd.DataFrame) -> pd.DataFrame:
    """Pareto-efficient methods among all nine at every (dataset, size), on two
    objective sets: (F1, CPU latency) for Fig. 3 and the full four-objective set."""
    rows = []
    for (d, n), cell in summary.groupby(["dataset", "size"]):
        pts = {m: _means(summary, d, n, m) for m in cell.method}
        for m, p in pts.items():
            two = all(not (o["test_macro_f1"] >= p["test_macro_f1"] and o["lat_cpu_e2e_ms"] <= p["lat_cpu_e2e_ms"]
                           and (o["test_macro_f1"] > p["test_macro_f1"] or o["lat_cpu_e2e_ms"] < p["lat_cpu_e2e_ms"]))
                      for k, o in pts.items() if k != m)
            four = all(not dominates(o, p) for k, o in pts.items() if k != m)
            rows.append({"dataset": d, "size": n, "method": m, "pareto_f1_latency": two, "pareto_all4": four})
    return pd.DataFrame(rows)


def marginal_gains(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-seed test macro-F1 gain (points) between adjacent sizes (nested subsets -> paired)."""
    sizes = sorted(int(s) for s in global_cfg()["sizes"])
    rows = []
    for (d, m), grp in runs.groupby(["dataset", "method"]):
        piv = grp.pivot_table(index="seed", columns="size", values="test_macro_f1")
        for a, b in zip(sizes[:-1], sizes[1:]):
            if a in piv and b in piv:
                g = ((piv[b] - piv[a]) * 100).dropna()
                rows.append({"dataset": d, "method": m, "from_size": a, "to_size": b, "gain_mean": g.mean(),
                             "gain_std": g.std(ddof=1) if len(g) > 1 else np.nan, "n_pairs": len(g)})
    return pd.DataFrame(rows)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    runs, summary = read_csv("runs.csv"), read_csv("summary.csv")
    outs = ("crossover_by_size.csv", "crossover.csv", "pareto_all.csv", "marginal_gains.csv")
    if runs is None or summary is None:
        print("no aggregated results yet - skipping crossover analysis")
        for f in outs:
            (AGG_DIR / f).unlink(missing_ok=True)
        return
    runs = runs[runs.tokenizer == main_tokenizer()]  # the ablation is analysed separately
    bs = pd.concat([by_size(runs, summary, "val"), by_size(runs, summary, "test")], ignore_index=True)
    bs.to_csv(AGG_DIR / "crossover_by_size.csv", index=False)
    if bs.empty:
        print("no (dataset, size) cell has both families yet")
        return
    ct = crossover_table(bs, summary)
    ct.to_csv(AGG_DIR / "crossover.csv", index=False)
    pareto_all(summary).to_csv(AGG_DIR / "pareto_all.csv", index=False)
    marginal_gains(runs).to_csv(AGG_DIR / "marginal_gains.csv", index=False)
    for _, r in ct.iterrows():
        thr = r.threshold if pd.notna(r.threshold) else "none <= 8000"
        print(f"[{r.dataset:8s}] selection={r.selection:4s} threshold={thr}  "
              f"{r.get('best_neural')} vs {r.get('best_tfidf')} gap={r.get('gap_mean', np.nan):.2f}"
              f"±{r.get('gap_std', np.nan):.2f}  pareto={r.get('neural_pareto_efficient')}")
    v = ct[ct.selection == "val"].set_index("dataset").threshold
    t = ct[ct.selection == "test"].set_index("dataset").threshold
    for d in v.index:
        if d in t.index and not (pd.isna(v[d]) and pd.isna(t[d])) and v[d] != t[d]:
            print(f"  sensitivity: {d} threshold differs (val-selected {v[d]} vs test-selected {t[d]})")


if __name__ == "__main__":
    main()
