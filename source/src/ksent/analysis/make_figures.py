"""Figures 2-5 and appendix Fig. A1 as PDF + PNG (300 dpi) into ``submission/report/figures/``.

Fig. 1 (pipeline) is a TikZ diagram (``figures/fig1_pipeline.tex``), not generated here.

Encoding: colour = method (fixed per method across every figure); TF-IDF methods
use dashed lines + hollow markers, neural methods solid lines + filled markers;
each method also has its own marker shape, so identity never relies on colour alone.

Usage:
  python -m ksent.analysis.make_figures                # from results/aggregated/
  python -m ksent.analysis.make_figures --placeholder  # dummy data + PLACEHOLDER watermark
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from ksent.analysis.common import FIG_DIR, METHOD_STYLE, datasets, dname, methods, mname, read_csv, sizes  # noqa: E402
from ksent.config import NEURAL_METHODS, family  # noqa: E402

ABBR = {"tfidf_lr": "LR", "tfidf_svm": "SVM", "tfidf_rf": "RF", "tfidf_xgb": "XGB", "emb_mlp": "MLP",
        "emb_lstm": "LSTM", "emb_gru": "GRU", "small_transformer": "Trf", "small_mamba": "Mamba"}
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e4e3dc"
BLUES = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "lines.linewidth": 1.4, "pdf.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150,
})


# ------------------------------------------------------------------ placeholder data
def dummy_summary(seed: int = 0) -> pd.DataFrame:
    """Plausible-looking fake summary used only for layout (watermarked)."""
    rng = np.random.default_rng(seed)
    rows = []
    for di, d in enumerate(datasets()):
        for mi, m in enumerate(methods()):
            base = 0.70 + 0.04 * di + 0.01 * (mi % 4)
            slope = 0.025 if family(m) == "tfidf" else 0.04
            lat = (0.4 + 0.3 * mi) if family(m) == "tfidf" else (0.8 + 1.5 * (mi - 3))
            for n in sizes():
                k = np.log2(n / 500)
                f1 = min(0.95, base + slope * k - (0.03 if family(m) == "neural" and k < 1.5 else 0))
                rows.append({"dataset": d, "method": m, "family": family(m), "size": n,
                             "test_macro_f1_mean": f1, "test_macro_f1_std": 0.004 + 0.01 * rng.random(),
                             "val_macro_f1_mean": f1 + 0.002, "val_macro_f1_std": 0.005,
                             "lat_cpu_e2e_ms_mean": lat, "lat_cpu_e2e_ms_std": 0.05 * lat,
                             "lat_acc_e2e_ms_mean": (lat * 1.6 if family(m) == "neural" else np.nan),
                             "lat_acc_e2e_ms_std": (0.05 * lat if family(m) == "neural" else np.nan)})
    return pd.DataFrame(rows)


def dummy_confusion(summary: pd.DataFrame) -> pd.DataFrame:
    n = max(sizes())
    s = summary[summary["size"] == n]
    rows = []
    for _, r in s.iterrows():
        tpr, tnr = r.test_macro_f1_mean + 0.02, r.test_macro_f1_mean - 0.02
        rows.append({"dataset": r.dataset, "method": r.method, "tnr": tnr, "fpr": 1 - tnr, "fnr": 1 - tpr, "tpr": tpr,
                     "n_runs": 5})
    return pd.DataFrame(rows)


def dummy_crossover() -> pd.DataFrame:
    return pd.DataFrame([{"dataset": d, "selection": "val", "threshold": t}
                         for d, t in zip(datasets(), [2000, 4000, np.nan])])


# ------------------------------------------------------------------ helpers
def watermark(fig) -> None:
    fig.text(0.5, 0.5, "PLACEHOLDER", fontsize=54, color="#e34948", alpha=0.22, rotation=20,
             ha="center", va="center", weight="bold", zorder=100)


def save(fig, name: str, placeholder: bool) -> None:
    if placeholder:
        watermark(fig)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf/.png{' (placeholder)' if placeholder else ''}")


def method_handles() -> list[Line2D]:
    hs = []
    for m in methods():
        st = METHOD_STYLE[m]
        hs.append(Line2D([], [], color=st["color"], linestyle=st["linestyle"], marker=st["marker"], markersize=4.5,
                         markerfacecolor=st["color"] if st["fill"] == "full" else "white", label=mname(m)))
    return hs


def _cell(summary, d, m, col):
    s = summary[(summary.dataset == d) & (summary.method == m)].sort_values("size")
    return s["size"].to_numpy(), s.get(f"{col}_mean", pd.Series(dtype=float)).to_numpy(), \
        s.get(f"{col}_std", pd.Series(np.zeros(len(s)))).fillna(0).to_numpy()


# ------------------------------------------------------------------ Fig. 2
def fig2_learning_curves(summary: pd.DataFrame, ct: pd.DataFrame | None, placeholder: bool) -> None:
    ds = datasets()
    fig, axes = plt.subplots(1, len(ds), figsize=(7.0, 2.55), sharey=False)
    for ax, d in zip(np.atleast_1d(axes), ds):
        for m in methods():
            x, y, e = _cell(summary, d, m, "test_macro_f1")
            if len(x) == 0:
                continue
            st = METHOD_STYLE[m]
            ax.fill_between(x, (y - e) * 100, (y + e) * 100, color=st["color"], alpha=0.12, linewidth=0)
            ax.plot(x, y * 100, color=st["color"], linestyle=st["linestyle"], marker=st["marker"], markersize=3.8,
                    markerfacecolor=st["color"] if st["fill"] == "full" else "white", markeredgewidth=1.0)
        if ct is not None:
            r = ct[(ct.dataset == d) & (ct.selection == "val")]
            if len(r) and pd.notna(r.threshold.iloc[0]):
                t = float(r.threshold.iloc[0])
                ax.axvline(t, color=INK2, linestyle=":", linewidth=1.0)
                ax.text(t, 1.0, " crossover", transform=ax.get_xaxis_transform(), fontsize=6.5, color=INK2,
                        va="top", ha="left")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes())
        ax.set_xticklabels([f"{n:,}" for n in sizes()])
        ax.minorticks_off()
        ax.set_title(dname(d), color=INK)
        ax.set_xlabel("Training reviews (log scale)")
    np.atleast_1d(axes)[0].set_ylabel("Test macro-F1 (×100)")
    fig.legend(handles=method_handles(), loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    save(fig, "fig2_learning_curves", placeholder)


# ------------------------------------------------------------------ Fig. 3
def pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Non-dominated points (min latency, max F1) sorted by latency."""
    front, best = [], -np.inf
    for x, y in sorted(points):
        if y > best:
            front.append((x, y))
            best = y
    return front


def fig3_quality_cost(summary: pd.DataFrame, placeholder: bool) -> None:
    n = max(sizes())
    s = summary[summary["size"] == n]
    ds = datasets()
    fig, axes = plt.subplots(1, len(ds), figsize=(7.0, 2.6))
    for ax, d in zip(np.atleast_1d(axes), ds):
        pts = []
        for m in methods():
            r = s[(s.dataset == d) & (s.method == m)]
            if r.empty or pd.isna(r.get("lat_cpu_e2e_ms_mean", pd.Series([np.nan])).iloc[0]):
                continue
            r = r.iloc[0]
            st = METHOD_STYLE[m]
            x, y = r.lat_cpu_e2e_ms_mean, r.test_macro_f1_mean * 100
            pts.append((x, y))
            face = st["color"] if st["fill"] == "full" else "white"
            if m in NEURAL_METHODS and pd.notna(r.get("lat_acc_e2e_ms_mean", np.nan)):
                gx = r.lat_acc_e2e_ms_mean
                ax.plot([x, gx], [y, y], color=st["color"], linewidth=0.8, alpha=0.6, zorder=2)
                ax.plot(gx, y, marker="*", markersize=6.5, color=st["color"], markeredgecolor="white",
                        markeredgewidth=0.5, linestyle="none", zorder=3)
            ax.errorbar(x, y, xerr=r.get("lat_cpu_e2e_ms_std", 0) or 0, yerr=(r.test_macro_f1_std or 0) * 100,
                        fmt=st["marker"], color=st["color"], markerfacecolor=face, markersize=4.5, elinewidth=0.8,
                        capsize=1.5, markeredgewidth=1.0, zorder=4)
            ax.annotate(ABBR[m], (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6, color=INK)
        front = pareto_front(pts)
        if front:
            fx, fy = zip(*front)
            ax.set_xscale("log")
            xmax = ax.get_xlim()[1] * 1.0
            step_x = list(fx) + [xmax]
            step_y = list(fy) + [fy[-1]]
            ymin = ax.get_ylim()[0]
            ax.fill_between(step_x, ymin, step_y, step="post", color=INK2, alpha=0.07, linewidth=0, zorder=0)
            ax.step(step_x, step_y, where="post", color=INK2, linewidth=0.8, linestyle="-", alpha=0.6, zorder=1)
            ax.set_ylim(bottom=ymin)
        ax.set_xscale("log")
        ax.set_title(dname(d), color=INK)
        ax.set_xlabel("End-to-end latency per review (ms, log)")
    np.atleast_1d(axes)[0].set_ylabel(f"Test macro-F1 (×100), n={n:,}")
    hs = method_handles() + [Line2D([], [], marker="*", color=INK2, linestyle="none", markersize=6.5,
                                    label="Accelerator (neural)"),
                             Line2D([], [], color=INK2, alpha=0.6, label="Pareto front (CPU)")]
    fig.legend(handles=hs, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    save(fig, "fig3_quality_vs_cost", placeholder)


# ------------------------------------------------------------------ Fig. 5 (tokenizer ablation)
TOK_LABEL = {"whitespace": "Eojeol", "kiwi": "Kiwi", "char": "Char"}


def dummy_ablation() -> pd.DataFrame:
    rows = []
    for di, d in enumerate(datasets()):
        for m, base in (("tfidf_svm", 0.78), ("emb_gru", 0.74)):
            for ti, t in enumerate(["whitespace", "kiwi", "char"]):
                for n in (500, 8000):
                    f1 = base + 0.03 * di + (0.06 if n == 8000 else 0) + (0.05 if m == "emb_gru" and n == 8000 else 0) \
                        + 0.01 * ti
                    rows.append({"dataset": d, "method": m, "tokenizer": t, "size": n, "test_macro_f1_mean": f1,
                                 "test_macro_f1_std": 0.006})
    return pd.DataFrame(rows)


def fig5_tokenizers(cells: pd.DataFrame, placeholder: bool) -> None:
    from ksent.config import global_cfg

    ab = global_cfg()["ablation"]
    toks, ns, ds = list(ab["tokenizers"]), [int(n) for n in ab["sizes"]], datasets()
    fig, axes = plt.subplots(len(ns), len(ds), figsize=(7.0, 3.9), squeeze=False, sharex=True)
    for i, n in enumerate(ns):
        for j, d in enumerate(ds):
            ax = axes[i, j]
            sub = cells[(cells.dataset == d) & (cells["size"] == n)]
            ms = list(dict.fromkeys(sub.method))
            for k, m in enumerate(ms):
                st = METHOD_STYLE[m]
                s = sub[sub.method == m].set_index("tokenizer").reindex(toks)
                x = np.arange(len(toks)) + (k - (len(ms) - 1) / 2) * 0.18
                ax.errorbar(x, s.test_macro_f1_mean * 100, yerr=s.test_macro_f1_std.fillna(0) * 100, fmt=st["marker"],
                            color=st["color"], markerfacecolor=st["color"] if st["fill"] == "full" else "white",
                            markersize=5, capsize=2, elinewidth=0.9, linestyle="none", label=mname(m))
            ax.set_xticks(range(len(toks)))
            ax.set_xticklabels([TOK_LABEL.get(t, t) for t in toks])
            ax.grid(axis="x", visible=False)
            if i == 0:
                ax.set_title(dname(d), color=INK)
            if j == 0:
                ax.set_ylabel(f"n={n:,}\nTest macro-F1 (×100)")
            if ms:
                ax.legend(loc="lower right", frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, "fig5_tokenizers", placeholder)


# ------------------------------------------------------------------ Fig. A1 (OOV vs size)
def figA1_oov(oov: pd.DataFrame, placeholder: bool) -> None:
    shades = ["#1f1f1e", "#5f5e58", "#9a998f"]
    markers = ["o", "s", "^"]
    k = oov[oov.tokenizer == "kiwi"]
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.4))
    for ax, col, title in zip(axes, ("oov_token_rate", "oov_review_rate"),
                              ("OOV tokens in test set", "Test reviews with ≥1 OOV token")):
        for i, d in enumerate(datasets()):
            g = k[k.dataset == d].groupby("size")[col].agg(["mean", "std"]).reset_index()
            if g.empty:
                continue
            ax.errorbar(g["size"], g["mean"] * 100, yerr=g["std"].fillna(0) * 100, color=shades[i], marker=markers[i],
                        markersize=4, capsize=1.5, linewidth=1.2, label=dname(d))
            ax.annotate(dname(d), (g["size"].iloc[-1], g["mean"].iloc[-1] * 100), xytext=(4, 0),
                        textcoords="offset points", fontsize=6, color=shades[i], va="center")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes())
        ax.set_xticklabels([f"{n:,}" for n in sizes()])
        ax.minorticks_off()
        ax.set_title(title, color=INK)
        ax.set_xlabel("Training reviews (log scale)")
        ax.set_ylabel("%")
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, "figA1_oov", placeholder)


def dummy_oov() -> pd.DataFrame:
    rows = []
    for i, d in enumerate(datasets()):
        for n in sizes():
            for s in range(3):
                r = 0.2 * (500 / n) ** 0.45 + 0.01 * i + 0.002 * s
                rows.append({"dataset": d, "tokenizer": "kiwi", "size": n, "seed": s, "oov_token_rate": r,
                             "oov_review_rate": min(1, 4.5 * r)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Fig. 4
def fig4_confusion(cm: pd.DataFrame, placeholder: bool) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    ds, ms = datasets(), methods()
    fig, axes = plt.subplots(len(ds), len(ms), figsize=(7.0, 2.75), squeeze=False)
    for i, d in enumerate(ds):
        for j, m in enumerate(ms):
            ax = axes[i, j]
            ax.grid(False)
            r = cm[(cm.dataset == d) & (cm.method == m)] if cm is not None else pd.DataFrame()
            if r.empty:
                ax.text(0.5, 0.5, "--", ha="center", va="center", transform=ax.transAxes, color=INK2)
                mat = None
            else:
                r = r.iloc[0]
                mat = np.array([[r.tnr, r.fpr], [r.fnr, r.tpr]])
                ax.imshow(mat, cmap=cmap, vmin=0, vmax=1)
                for a in range(2):
                    for b in range(2):
                        ax.text(b, a, f"{mat[a, b] * 100:.0f}", ha="center", va="center", fontsize=6,
                                color="white" if mat[a, b] > 0.6 else INK)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["neg", "pos"] if i == len(ds) - 1 else [], fontsize=5.5)
            ax.set_yticklabels(["neg", "pos"] if j == 0 else [], fontsize=5.5)
            ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(ABBR[m], fontsize=7, color=INK)
            if j == 0:
                ax.set_ylabel(dname(d), fontsize=7, color=INK)
    fig.text(0.5, -0.01, "Predicted label", ha="center", fontsize=7, color=INK2)
    fig.text(-0.01, 0.5, "True label", va="center", rotation=90, fontsize=7, color=INK2)
    fig.tight_layout(w_pad=0.3, h_pad=0.4)
    save(fig, "fig4_confusion", placeholder)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--placeholder", action="store_true")
    args = ap.parse_args()
    summary, ct, cm = read_csv("summary.csv"), read_csv("crossover.csv"), read_csv("confusion_n8000.csv")
    placeholder = args.placeholder or summary is None
    if placeholder:
        if not args.placeholder:
            print("no results/aggregated/summary.csv - drawing placeholder figures")
        summary = dummy_summary()
        ct, cm = dummy_crossover(), dummy_confusion(summary)
    fig2_learning_curves(summary, ct, placeholder)
    fig3_quality_cost(summary, placeholder)
    fig4_confusion(cm, placeholder)
    cells = None if args.placeholder else read_csv("ablation_cells.csv")
    fig5_tokenizers(cells if cells is not None else dummy_ablation(), cells is None)
    oov = None if args.placeholder else read_csv("diagnostics_oov.csv")
    figA1_oov(oov if oov is not None else dummy_oov(), oov is None)


if __name__ == "__main__":
    main()
