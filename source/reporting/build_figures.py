from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reporting_utils import (
    COLORS,
    DATASETS,
    DISPLAY,
    MARKERS,
    NEURAL_MODELS,
    REPORTING_RESULTS,
    RESULTS,
    SUBMISSION,
    TFIDF_MODELS,
    write_json,
)


DERIVED = REPORTING_RESULTS / "derived_metrics"
FIGURE_DATA = REPORTING_RESULTS / "figure_data"
OUT = SUBMISSION / "assets" / "figures"


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def dataset_overview(manifest: dict) -> None:
    frame = pd.read_csv(DERIVED / "dataset_statistics.csv")
    data = frame[
        [
            "dataset", "train_negative", "train_positive", "validation_negative",
            "validation_positive", "test_negative", "test_positive",
        ]
    ]
    data.to_csv(FIGURE_DATA / "dataset_overview.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.15), sharey=False)
    for ax, (_, row) in zip(axes, frame.iterrows()):
        splits = ["Train", "Validation", "Test"]
        negative = [row.train_negative, row.validation_negative, row.test_negative]
        positive = [row.train_positive, row.validation_positive, row.test_positive]
        x = np.arange(3)
        ax.bar(x, negative, color="#0072B2", label="Negative")
        ax.bar(x, positive, bottom=negative, color="#E69F00", label="Positive")
        ax.set_xticks(x, splits)
        ax.set_title(DISPLAY[row.dataset])
        ax.set_ylabel("Reviews")
        ax.ticklabel_format(axis="y", style="plain")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Cleaned splits remain nearly class-balanced", y=1.02, fontweight="semibold")
    fig.tight_layout()
    _save(fig, "dataset_overview")
    manifest["dataset_overview"] = {
        "data": "Source/results/reporting/figure_data/dataset_overview.csv",
        "source": "Source/data/splits/*_splits.csv",
        "rule": "count labels by dataset and fixed split",
    }


def part1_learning_curves(manifest: dict) -> None:
    frame = pd.read_csv(DERIVED / "part1_summary.csv")
    frame.to_csv(FIGURE_DATA / "part1_learning_curves.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), sharey=True)
    models = [*TFIDF_MODELS, *NEURAL_MODELS]
    for ax, dataset in zip(axes, DATASETS):
        group = frame[frame.dataset.eq(dataset)]
        for model in models:
            curve = group[group.model.eq(model)].sort_values("train_size")
            ax.errorbar(
                curve.train_size,
                curve.macro_f1_mean,
                yerr=curve.macro_f1_sd,
                label=DISPLAY[model],
                color=COLORS[model],
                marker=MARKERS[model],
                markersize=4,
                linewidth=1.2,
                capsize=1.5,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(group.train_size.unique()[::2])
        ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda value, _: f"{int(value/1000)}k" if value >= 1000 else f"{int(value)}"))
        ax.set_title(DISPLAY[dataset])
        ax.set_xlabel("Labeled training examples")
        ax.set_ylim(0.45, 0.92)
    axes[0].set_ylabel("Test macro-F1")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Model rankings change as the labeled set grows", y=1.02, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    _save(fig, "part1_learning_curves")
    manifest["part1_learning_curves"] = {
        "data": "Source/results/reporting/figure_data/part1_learning_curves.csv",
        "source": "Source/results/reporting/derived_metrics/part1_summary.csv",
        "rule": "mean with sample-SD error bars across three final seeds",
    }


def part1_crossover(manifest: dict) -> None:
    frame = pd.read_csv(DERIVED / "part1_best_by_size.csv")
    frame.to_csv(FIGURE_DATA / "part1_crossover.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.45), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        group = frame[frame.dataset.eq(dataset)].sort_values("train_size")
        ax.axhline(0.01, color="#D55E00", linestyle="--", linewidth=1, label="0.01 criterion")
        ax.axhline(0, color="#555555", linewidth=0.7)
        ax.plot(group.train_size, group.neural_minus_tfidf, color="#000000", marker="o", linewidth=1.4)
        above = group[group.neural_minus_tfidf >= 0.01]
        ax.scatter(above.train_size, above.neural_minus_tfidf, color="#009E73", s=30, zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(group.train_size.unique()[::2])
        ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda value, _: f"{int(value/1000)}k" if value >= 1000 else f"{int(value)}"))
        ax.set_title(DISPLAY[dataset])
        ax.set_xlabel("Training examples")
    axes[0].set_ylabel("Best neural minus best TF-IDF macro-F1")
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("A sustained 0.01 crossover appears only on NSMC", y=1.02, fontweight="semibold")
    fig.tight_layout()
    _save(fig, "part1_crossover")
    manifest["part1_crossover"] = {
        "data": "Source/results/reporting/figure_data/part1_crossover.csv",
        "source": "Source/results/reporting/derived_metrics/part1_best_by_size.csv",
        "rule": "best complete neural mean minus best complete TF-IDF mean at each size",
    }


def full_data_comparison(manifest: dict) -> None:
    frame = pd.read_csv(DERIVED / "part1_full_data_ranking.csv")
    frame.to_csv(FIGURE_DATA / "full_data_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.55), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        group = frame[frame.dataset.eq(dataset)].sort_values("macro_f1_mean")
        y = np.arange(len(group))
        ax.barh(y, group.macro_f1_mean, xerr=group.macro_f1_sd, color=[COLORS[model] for model in group.model], capsize=2)
        ax.set_yticks(y, [DISPLAY[model] for model in group.model])
        ax.set_xlim(0.65, 0.91)
        ax.set_title(DISPLAY[dataset])
        ax.set_xlabel("Test macro-F1")
    fig.suptitle("Neural models lead at full data, with small gaps among the top methods", y=1.02, fontweight="semibold")
    fig.tight_layout()
    _save(fig, "full_data_comparison")
    manifest["full_data_comparison"] = {
        "data": "Source/results/reporting/figure_data/full_data_comparison.csv",
        "source": "Source/results/reporting/derived_metrics/part1_full_data_ranking.csv",
        "rule": "full-data mean and sample SD across three final seeds",
    }


def efficiency_pareto(manifest: dict) -> None:
    frame = pd.read_csv(DERIVED / "part1_efficiency_full_data.csv")
    frame.to_csv(FIGURE_DATA / "efficiency_pareto.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        group = frame[frame.dataset.eq(dataset)]
        for _, row in group.iterrows():
            ax.scatter(
                row.fit_time_sec_mean,
                row.macro_f1_mean,
                color=COLORS[row.model],
                marker=MARKERS[row.model],
                s=60 if row.pareto_training else 36,
                edgecolor="black" if row.pareto_training else "none",
                linewidth=0.7,
            )
            ax.annotate(DISPLAY[row.model].replace("TF-IDF + ", ""), (row.fit_time_sec_mean, row.macro_f1_mean), xytext=(3, 2), textcoords="offset points", fontsize=6.5)
        ax.set_xscale("log")
        ax.set_title(DISPLAY[dataset])
        ax.set_xlabel("Mean fit time (s, log scale)")
    axes[0].set_ylabel("Full-data macro-F1")
    fig.suptitle("Higher full-data quality comes with sharply different training costs", y=1.02, fontweight="semibold")
    fig.tight_layout()
    _save(fig, "efficiency_pareto")
    manifest["efficiency_pareto"] = {
        "data": "Source/results/reporting/figure_data/efficiency_pareto.csv",
        "source": "Source/results/reporting/derived_metrics/part1_efficiency_full_data.csv",
        "rule": "mean across three seeds; outlined points are non-dominated on quality versus fit time",
    }


def bonus_learning_curves(manifest: dict) -> None:
    frame = pd.read_csv(DERIVED / "bonus_summary_corrected.csv")
    frame.to_csv(FIGURE_DATA / "bonus_learning_curves.csv", index=False)
    methods = ["best_tfidf", "target_transformer", "pooled_ce", "supcon", "db_cdsupcon"]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), sharey=True)
    for ax, target in zip(axes, DATASETS):
        group = frame[frame.target_domain.eq(target)]
        for method in methods:
            curve = group[group.method.eq(method)].sort_values("target_train_size")
            ax.errorbar(
                curve.target_train_size,
                curve.target_macro_f1_mean,
                yerr=curve.target_macro_f1_sd,
                label=DISPLAY[method],
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=1.4,
                capsize=2,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks([200, 500, 1000, 2000], ["200", "500", "1k", "2k"])
        ax.set_title(DISPLAY[target])
        ax.set_xlabel("New target-domain labels")
    axes[0].set_ylabel("Target test macro-F1")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Pooling helps the target-only Transformer, but TF-IDF remains difficult to beat", y=1.02, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    _save(fig, "bonus_learning_curves")
    manifest["bonus_learning_curves"] = {
        "data": "Source/results/reporting/figure_data/bonus_learning_curves.csv",
        "source": "Source/results/reporting/derived_metrics/bonus_summary_corrected.csv",
        "rule": "fixed Part 1 TF-IDF model per target/size; mean and sample SD across aligned seeds",
    }


def bonus_crossover_status(manifest: dict) -> None:
    frame = pd.read_csv(DERIVED / "bonus_label_efficiency_corrected.csv")
    frame.to_csv(FIGURE_DATA / "bonus_crossover_status.csv", index=False)
    methods = ["target_transformer", "pooled_ce", "supcon", "db_cdsupcon"]
    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    ax.set_xlim(-0.5, len(methods) - 0.5)
    ax.set_ylim(-0.5, len(DATASETS) - 0.5)
    for y, target in enumerate(DATASETS):
        for x, method in enumerate(methods):
            row = frame[frame.target_domain.eq(target) & frame.method.eq(method)].iloc[0]
            observed = pd.notna(row.crossover_threshold_size)
            ax.scatter(x, y, s=950, marker="s", color="#DDEBF2" if observed else "#EEEEEE", edgecolor="#555555", linewidth=0.7)
            text = f"{int(row.crossover_threshold_size):,}" if observed else "N/O"
            ax.text(x, y, text, ha="center", va="center", fontsize=9, fontweight="semibold" if observed else "normal")
    ax.set_xticks(range(len(methods)), [DISPLAY[method] for method in methods], rotation=15, ha="right")
    ax.set_yticks(range(len(DATASETS)), [DISPLAY[target] for target in DATASETS])
    ax.grid(False)
    ax.set_title("Crossover against best TF-IDF within 200–2,000 target labels")
    ax.text(0.5, -0.36, "N/O = not observed; target-only crossover is absent, so label savings are not estimable", transform=ax.transAxes, ha="center", fontsize=8)
    fig.tight_layout()
    _save(fig, "bonus_crossover_status")
    manifest["bonus_crossover_status"] = {
        "data": "Source/results/reporting/figure_data/bonus_crossover_status.csv",
        "source": "Source/results/reporting/derived_metrics/bonus_label_efficiency_corrected.csv",
        "rule": "first target size with method mean at least 0.01 above fixed best-TF-IDF mean",
    }


def bonus_transfer(manifest: dict) -> None:
    summary = pd.read_csv(DERIVED / "bonus_summary_corrected.csv")
    pooled = summary[summary.method.eq("pooled_ce")][
        ["target_domain", "target_train_size", "target_macro_f1_mean", "mean_source_macro_f1_mean"]
    ].rename(columns={"target_macro_f1_mean": "pooled_target", "mean_source_macro_f1_mean": "pooled_source"})
    frame = summary[summary.method.isin(["supcon", "db_cdsupcon"])].merge(pooled, on=["target_domain", "target_train_size"])
    frame["target_difference"] = frame.target_macro_f1_mean - frame.pooled_target
    frame["source_difference"] = frame.mean_source_macro_f1_mean - frame.pooled_source
    frame.to_csv(FIGURE_DATA / "bonus_transfer.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.3), sharex=True, sharey=True)
    for ax, target in zip(axes, DATASETS):
        group = frame[frame.target_domain.eq(target)]
        for method in ["supcon", "db_cdsupcon"]:
            cells = group[group.method.eq(method)]
            ax.scatter(
                cells.source_difference,
                cells.target_difference,
                color=COLORS[method],
                marker=MARKERS[method],
                s=45,
                label=DISPLAY[method],
            )
            for _, row in cells.iterrows():
                ax.annotate(str(int(row.target_train_size)), (row.source_difference, row.target_difference), xytext=(3, 2), textcoords="offset points", fontsize=6.5)
        ax.axhline(0, color="#777777", linewidth=0.7)
        ax.axvline(0, color="#777777", linewidth=0.7)
        ax.set_title(DISPLAY[target])
        ax.set_xlabel("Source macro-F1 difference vs pooled CE")
    axes[0].set_ylabel("Target macro-F1 difference vs pooled CE")
    axes[0].legend(frameon=False)
    fig.suptitle("Contrastive effects on target and source performance vary by domain and label count", y=1.02, fontweight="semibold")
    fig.tight_layout()
    _save(fig, "bonus_transfer")
    manifest["bonus_transfer"] = {
        "data": "Source/results/reporting/figure_data/bonus_transfer.csv",
        "source": "Source/results/reporting/derived_metrics/bonus_summary_corrected.csv",
        "rule": "method mean minus matched pooled-CE mean; labels are target-label counts",
    }


def bonus_representation(manifest: dict) -> None:
    frame = pd.read_csv(DERIVED / "bonus_representation_diagnostics.csv")
    frame.to_csv(FIGURE_DATA / "bonus_representation.csv", index=False)
    methods = ["pooled_ce", "supcon", "db_cdsupcon"]
    aggregate = frame.groupby(["target_domain", "method"]).agg(
        mean=("separation_gap_mean", "mean"),
        sd=("separation_gap_mean", "std"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(7.8, 3.3))
    width = 0.23
    x = np.arange(len(DATASETS))
    for offset, method in enumerate(methods):
        group = aggregate[aggregate.method.eq(method)].set_index("target_domain").loc[list(DATASETS)]
        ax.bar(
            x + (offset - 1) * width,
            group["mean"],
            width,
            yerr=group.sd,
            color=COLORS[method],
            label=DISPLAY[method],
            capsize=2,
        )
    ax.set_xticks(x, [DISPLAY[target] for target in DATASETS])
    ax.set_ylabel("Cross-domain sentiment separation gap")
    ax.legend(frameon=False, ncol=3)
    ax.set_title("Representation separation changes do not translate uniformly into target gains")
    fig.tight_layout()
    _save(fig, "bonus_representation")
    manifest["bonus_representation"] = {
        "data": "Source/results/reporting/figure_data/bonus_representation.csv",
        "source": "Source/results/reporting/derived_metrics/bonus_representation_diagnostics.csv",
        "rule": "mean over three seeds, then mean and SD across four target-label sizes",
    }


def confusion_comparison(manifest: dict) -> None:
    summary = pd.read_csv(DERIVED / "part1_full_data_ranking.csv")
    records = pd.read_csv(RESULTS / "csv" / "run_records.csv", keep_default_na=False)
    dataset = "nsmc"
    group = summary[summary.dataset.eq(dataset)]
    tfidf = group[group.model.isin(TFIDF_MODELS)].sort_values("macro_f1_mean", ascending=False).iloc[0]
    neural = group[group.model.isin(NEURAL_MODELS)].sort_values("macro_f1_mean", ascending=False).iloc[0]
    rows = []
    matrices = []
    for model in [tfidf.model, neural.model]:
        selected = records[
            records.phase.eq("part1_final")
            & records.dataset.eq(dataset)
            & records.model.eq(model)
            & pd.to_numeric(records.train_size).eq(int(tfidf.train_size))
        ]
        matrix = np.array(
            [
                [
                    pd.to_numeric(selected.tn).sum(),
                    pd.to_numeric(selected.fp).sum(),
                ],
                [
                    pd.to_numeric(selected.fn).sum(),
                    pd.to_numeric(selected.tp).sum(),
                ],
            ],
            dtype=float,
        )
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        matrices.append((model, matrix, normalized))
        for actual in range(2):
            for predicted in range(2):
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "actual": actual,
                        "predicted": predicted,
                        "count_three_seeds": int(matrix[actual, predicted]),
                        "row_normalized": normalized[actual, predicted],
                        "run_ids": ",".join(selected.run_id),
                    }
                )
    pd.DataFrame(rows).to_csv(FIGURE_DATA / "confusion_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1))
    for ax, (model, matrix, normalized) in zip(axes, matrices):
        image = ax.imshow(normalized, cmap=mpl.colors.LinearSegmentedColormap.from_list("blue", ["#FFFFFF", "#0072B2"]), vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{normalized[i,j]:.1%}\n({int(matrix[i,j]):,})", ha="center", va="center", color="white" if normalized[i,j] > 0.55 else "black", fontsize=9)
        ax.set_xticks([0, 1], ["Negative", "Positive"])
        ax.set_yticks([0, 1], ["Negative", "Positive"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(DISPLAY[model])
        ax.grid(False)
    fig.suptitle("NSMC full-data errors: best TF-IDF versus best neural method", y=1.02, fontweight="semibold")
    fig.tight_layout()
    _save(fig, "confusion_comparison")
    manifest["confusion_comparison"] = {
        "data": "Source/results/reporting/figure_data/confusion_comparison.csv",
        "source": "Source/results/csv/run_records.csv",
        "rule": "sum three seed confusion counts, then normalize within actual class",
    }


def main() -> None:
    _style()
    FIGURE_DATA.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, str]] = {}
    dataset_overview(manifest)
    part1_learning_curves(manifest)
    part1_crossover(manifest)
    full_data_comparison(manifest)
    efficiency_pareto(manifest)
    bonus_learning_curves(manifest)
    bonus_crossover_status(manifest)
    bonus_transfer(manifest)
    bonus_representation(manifest)
    confusion_comparison(manifest)
    write_json(REPORTING_RESULTS / "figure_manifest.json", manifest)
    print(f"built {len(manifest)} figures in PDF and PNG")


if __name__ == "__main__":
    main()
