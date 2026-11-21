from __future__ import annotations

import itertools
import json
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from reporting_utils import (
    DATASETS,
    DISPLAY,
    FINAL_SEEDS,
    NEURAL_MODELS,
    REPORTING_RESULTS,
    RESULTS,
    SOURCE,
    SUBMISSION,
    TFIDF_MODELS,
    best_complete,
    first_sustained_threshold,
    first_threshold,
    format_mean_sd,
    labels_saved,
    latex_escape,
    mean_sd,
    normalized_log_auc,
    pareto_mask,
    write_csv,
    write_json,
)


DERIVED = REPORTING_RESULTS / "derived_metrics"
FIGURE_DATA = REPORTING_RESULTS / "figure_data"
TABLES = SUBMISSION / "assets" / "tables"


class Trace:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.ids: set[str] = set()

    def add(
        self,
        value_id: str,
        value: Any,
        display: str,
        source_file: str,
        source_rows: str,
        derivation: str,
        artifact: str,
    ) -> str:
        if value_id in self.ids:
            return value_id
        self.ids.add(value_id)
        self.rows.append(
            {
                "value_id": value_id,
                "value": value,
                "display": display,
                "source_file": source_file,
                "source_rows": source_rows,
                "derivation": derivation,
                "artifact": artifact,
            }
        )
        return value_id


def _records() -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / "csv" / "run_records.csv", keep_default_na=False)
    return frame[frame.status.eq("completed")].copy()


def _dataset_stats(trace: Trace) -> pd.DataFrame:
    base = pd.read_csv(RESULTS / "csv" / "dataset_statistics.csv")
    rows = []
    for dataset in DATASETS:
        split = pd.read_csv(SOURCE / "data" / "splits" / f"{dataset}_splits.csv")
        current = base[base.dataset.eq(dataset)].iloc[0]
        row: dict[str, Any] = {
            "dataset": dataset,
            "train_rows": int(current.train_rows),
            "validation_rows": int(current.validation_rows),
            "test_rows": int(current.test_rows),
            "n_full": int(current.n_full),
            "training_sizes": current.training_sizes,
        }
        for name in ("train", "validation", "test"):
            counts = split[split.split.eq(name)].label.value_counts()
            row[f"{name}_negative"] = int(counts.get(0, 0))
            row[f"{name}_positive"] = int(counts.get(1, 0))
            row[f"{name}_positive_pct"] = 100 * row[f"{name}_positive"] / (
                row[f"{name}_negative"] + row[f"{name}_positive"]
            )
        report = json.loads((SOURCE / "data" / "manifests" / f"{dataset}_cleaning_report.json").read_text())
        row["raw_rows"] = sum(value["raw_rows"] for value in report["sources"].values())
        row["clean_source_rows"] = sum(value["clean_rows"] for value in report["sources"].values())
        row["removed_rows"] = row["raw_rows"] - row["clean_source_rows"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(DERIVED / "dataset_statistics.csv", index=False)
    for row in rows:
        for field in ("train_rows", "validation_rows", "test_rows"):
            trace.add(
                f"dataset_{row['dataset']}_{field}",
                row[field],
                f"{row[field]:,}",
                "source/results/csv/dataset_statistics.csv",
                row["dataset"],
                "direct field",
                "table_dataset_stats",
            )
    return frame


def _part1(
    records: pd.DataFrame, trace: Trace
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = records[records.phase.eq("part1_final")].copy()
    numeric = [
        "train_size", "seed", "test_macro_f1", "test_accuracy", "test_macro_precision", "test_macro_recall",
        "fit_time_sec", "duration_sec", "classifier_latency_ms_p50", "end_to_end_latency_ms_p50",
        "throughput_samples_per_sec", "peak_host_ram_mb", "peak_gpu_memory_mb", "serialized_size_mb",
        "total_parameters", "trainable_parameters", "tree_count", "node_count", "vocab_size", "feature_count",
    ]
    for column in numeric:
        seed[column] = pd.to_numeric(seed[column], errors="coerce")
    seed = seed.sort_values(["dataset", "model", "train_size", "seed"])
    seed.to_csv(DERIVED / "part1_metrics_by_seed.csv", index=False)

    summary_rows = []
    for (dataset, model, size), group in seed.groupby(["dataset", "model", "train_size"], sort=True):
        mean, sd = mean_sd(group.test_macro_f1.tolist())
        summary_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "model_family": group.model_family.iloc[0],
                "train_size": int(size),
                "macro_f1_mean": mean,
                "macro_f1_sd": sd,
                "n": group.seed.nunique(),
                "complete": group.seed.nunique() == len(FINAL_SEEDS),
                "run_ids": ",".join(group.run_id),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["dataset", "model", "train_size"])
    summary["gain_previous"] = np.nan
    summary["approximately_half_size"] = np.nan
    summary["gain_approximately_doubled"] = np.nan
    summary["normalized_log2_aulc"] = np.nan
    for indices in summary.groupby(["dataset", "model"]).groups.values():
        curve = summary.loc[indices].sort_values("train_size")
        summary.loc[curve.index, "normalized_log2_aulc"] = normalized_log_auc(
            curve.train_size.tolist(), curve.macro_f1_mean.tolist()
        )
        for position, (index, row) in enumerate(curve.iterrows()):
            if position:
                previous = curve.iloc[position - 1]
                summary.loc[index, "gain_previous"] = row.macro_f1_mean - previous.macro_f1_mean
                candidates = curve.iloc[:position][curve.iloc[:position].train_size <= row.train_size / 2]
                if not candidates.empty:
                    half = candidates.iloc[-1]
                    summary.loc[index, "approximately_half_size"] = half.train_size
                    summary.loc[index, "gain_approximately_doubled"] = row.macro_f1_mean - half.macro_f1_mean
    summary.to_csv(DERIVED / "part1_summary.csv", index=False)

    best_rows = []
    crossover_rows = []
    for dataset, dataset_rows in summary.groupby("dataset"):
        cells = []
        for size, group in dataset_rows.groupby("train_size"):
            tfidf = pd.Series(best_complete(group[group.model_family == "tfidf"].to_dict("records"), "macro_f1_mean"))
            neural = pd.Series(best_complete(group[group.model_family == "neural"].to_dict("records"), "macro_f1_mean"))
            cells.append(
                {
                    "dataset": dataset,
                    "train_size": int(size),
                    "best_tfidf_model": tfidf.model,
                    "best_tfidf_mean": tfidf.macro_f1_mean,
                    "best_tfidf_sd": tfidf.macro_f1_sd,
                    "best_neural_model": neural.model,
                    "best_neural_mean": neural.macro_f1_mean,
                    "best_neural_sd": neural.macro_f1_sd,
                    "neural_minus_tfidf": neural.macro_f1_mean - tfidf.macro_f1_mean,
                    "criterion_met": neural.macro_f1_mean - tfidf.macro_f1_mean >= 0.01,
                    "tfidf_run_ids": tfidf.run_ids,
                    "neural_run_ids": neural.run_ids,
                }
            )
        cells = sorted(cells, key=lambda row: row["train_size"])
        observed = first_threshold(
            [row["train_size"] for row in cells], [row["neural_minus_tfidf"] for row in cells]
        )
        sustained = first_sustained_threshold(
            [row["train_size"] for row in cells], [row["neural_minus_tfidf"] for row in cells]
        )
        max_size = cells[-1]["train_size"]
        crossover_rows.append(
            {
                "dataset": dataset,
                "first_observed_crossover": observed,
                "first_sustained_crossover": sustained,
                "status": "observed_and_sustained" if sustained is not None else f"not_observed_through_{max_size}",
                "largest_tested_size": max_size,
                "sustained_definition": (
                    "neural-minus-TF-IDF mean macro-F1 is at least 0.01 at this and every larger tested size"
                ),
            }
        )
        for row in cells:
            row["first_observed_crossover"] = observed
            row["first_sustained_crossover"] = sustained
        best_rows.extend(cells)
    best = pd.DataFrame(best_rows)
    crossovers = pd.DataFrame(crossover_rows)
    best.to_csv(DERIVED / "part1_best_by_size.csv", index=False)
    crossovers.to_csv(DERIVED / "part1_crossover_summary.csv", index=False)

    full_sizes = summary.groupby("dataset").train_size.max().to_dict()
    full = summary[
        summary.apply(lambda row: row.train_size == full_sizes[row.dataset], axis=1)
    ].copy()
    full["rank"] = full.groupby("dataset").macro_f1_mean.rank(method="min", ascending=False).astype(int)
    full = full.sort_values(["dataset", "rank", "model"])
    full.to_csv(DERIVED / "part1_full_data_ranking.csv", index=False)

    for row in crossover_rows:
        display = (
            f"{int(row['first_sustained_crossover']):,}"
            if row["first_sustained_crossover"] is not None
            else f"not observed through {int(row['largest_tested_size']):,}"
        )
        trace.add(
            f"part1_crossover_{row['dataset']}",
            row["first_sustained_crossover"],
            display,
            "source/results/reporting/derived_metrics/part1_best_by_size.csv",
            row["dataset"],
            "first size with best-neural minus best-TF-IDF mean macro-F1 >= 0.01 at all larger tested sizes",
            "table_part1_crossover,report,slides",
        )
    return seed, summary, best, full


def _efficiency(seed: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    sizes = full.set_index("dataset").train_size.to_dict()
    rows = []
    metrics = [
        "fit_time_sec", "classifier_latency_ms_p50", "end_to_end_latency_ms_p50",
        "throughput_samples_per_sec", "peak_host_ram_mb", "peak_gpu_memory_mb",
        "serialized_size_mb", "total_parameters", "tree_count", "node_count",
    ]
    for (dataset, model), group in seed.groupby(["dataset", "model"]):
        group = group[group.train_size.eq(sizes[dataset])]
        row: dict[str, Any] = {
            "dataset": dataset, "model": model, "n": group.seed.nunique(), "run_ids": ",".join(group.run_id)
        }
        for metric in metrics:
            values = group[metric].dropna()
            row[f"{metric}_mean"] = values.mean() if len(values) else np.nan
            row[f"{metric}_max"] = values.max() if len(values) else np.nan
        row["macro_f1_mean"] = group.test_macro_f1.mean()
        row["macro_f1_sd"] = group.test_macro_f1.std(ddof=1)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["pareto_training"] = False
    frame["pareto_inference"] = False
    for indices in frame.groupby("dataset").groups.values():
        group = frame.loc[indices]
        frame.loc[indices, "pareto_training"] = pareto_mask(group.macro_f1_mean, group.fit_time_sec_mean)
        frame.loc[indices, "pareto_inference"] = pareto_mask(
            group.macro_f1_mean, group.end_to_end_latency_ms_p50_mean
        )
    frame.to_csv(DERIVED / "part1_efficiency_full_data.csv", index=False)
    return frame


def _bonus(
    records: pd.DataFrame,
    part1_seed: pd.DataFrame,
    part1_summary: pd.DataFrame,
    trace: Trace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_sizes = (200, 500, 1000, 2000)
    rows = []
    fixed_tfidf: dict[tuple[str, int], str] = {}
    for dataset in DATASETS:
        for size in target_sizes:
            candidates = part1_summary[
                part1_summary.dataset.eq(dataset)
                & part1_summary.train_size.eq(size)
                & part1_summary.model.isin(TFIDF_MODELS)
            ]
            fixed_tfidf[(dataset, size)] = candidates.sort_values(
                ["macro_f1_mean", "model"], ascending=[False, True]
            ).iloc[0].model
            base = part1_seed[part1_seed.dataset.eq(dataset) & part1_seed.train_size.eq(size)]
            for method, model in (
                ("best_tfidf", fixed_tfidf[(dataset, size)]),
                ("target_transformer", "transformer"),
            ):
                for _, row in base[base.model.eq(model)].iterrows():
                    rows.append(
                        {
                            "target_domain": dataset,
                            "target_train_size": size,
                            "seed": int(row.seed),
                            "method": method,
                            "target_macro_f1": row.test_macro_f1,
                            "target_accuracy": row.test_accuracy,
                            "mean_source_macro_f1": np.nan,
                            "all_domain_macro_f1": np.nan,
                            "run_id": row.run_id,
                            "record_origin": "reused_part1",
                            "reference_model": model,
                        }
                    )
    fresh = records[records.phase.eq("bonus_final")].copy()
    for column in [
        "target_train_size", "seed", "target_test_macro_f1", "test_accuracy",
        "mean_source_test_macro_f1", "all_domain_mean_macro_f1", "fit_time_sec",
        "classifier_latency_ms_p50", "end_to_end_latency_ms_p50", "peak_gpu_memory_mb",
        "peak_host_ram_mb", "serialized_size_mb", "deployed_parameter_count",
        "projection_parameters", "same_sentiment_cross_domain_cosine",
        "different_sentiment_cross_domain_cosine", "cross_domain_sentiment_separation_gap",
        "same_domain_same_sentiment_cosine",
    ]:
        fresh[column] = pd.to_numeric(fresh[column], errors="coerce")
    for _, row in fresh.iterrows():
        rows.append(
            {
                "target_domain": row.target_domain,
                "target_train_size": int(row.target_train_size),
                "seed": int(row.seed),
                "method": row.objective,
                "target_macro_f1": row.target_test_macro_f1,
                "target_accuracy": row.test_accuracy,
                "mean_source_macro_f1": row.mean_source_test_macro_f1,
                "all_domain_macro_f1": row.all_domain_mean_macro_f1,
                "run_id": row.run_id,
                "record_origin": "new_bonus",
                "reference_model": "transformer",
            }
        )
    seed = pd.DataFrame(rows).sort_values(
        ["target_domain", "method", "target_train_size", "seed"]
    )
    seed.to_csv(DERIVED / "bonus_metrics_by_seed_corrected.csv", index=False)

    summary_rows = []
    for (target, method, size), group in seed.groupby(
        ["target_domain", "method", "target_train_size"]
    ):
        mean, sd = mean_sd(group.target_macro_f1.tolist())
        source_mean, source_sd = mean_sd(group.mean_source_macro_f1.dropna().tolist())
        summary_rows.append(
            {
                "target_domain": target,
                "method": method,
                "target_train_size": int(size),
                "target_macro_f1_mean": mean,
                "target_macro_f1_sd": sd,
                "mean_source_macro_f1_mean": source_mean,
                "mean_source_macro_f1_sd": source_sd,
                "n": group.seed.nunique(),
                "complete": group.seed.nunique() == len(FINAL_SEEDS),
                "run_ids": ",".join(group.run_id),
                "reference_model": group.reference_model.iloc[0],
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["target_domain", "method", "target_train_size"]
    )
    pooled_source = summary[summary.method.eq("pooled_ce")].set_index(
        ["target_domain", "target_train_size"]
    ).mean_source_macro_f1_mean
    summary["source_retention_vs_pooled_ce"] = summary.apply(
        lambda row: np.nan
        if pd.isna(row.mean_source_macro_f1_mean)
        else row.mean_source_macro_f1_mean
        - pooled_source[(row.target_domain, row.target_train_size)],
        axis=1,
    )
    summary.to_csv(DERIVED / "bonus_summary_corrected.csv", index=False)

    pair_rows = []
    methods = ["target_transformer", "pooled_ce", "supcon", "db_cdsupcon"]
    for target, size in itertools.product(DATASETS, target_sizes):
        group = seed[
            seed.target_domain.eq(target) & seed.target_train_size.eq(size)
        ]
        pivot = group.pivot(index="seed", columns="method", values="target_macro_f1")
        run_pivot = group.pivot(index="seed", columns="method", values="run_id")
        for left, right in itertools.permutations(methods, 2):
            differences = (pivot[left] - pivot[right]).dropna()
            mean, sd = mean_sd(differences.tolist())
            pair_rows.append(
                {
                    "target_domain": target,
                    "target_train_size": size,
                    "left_method": left,
                    "right_method": right,
                    "mean_difference": mean,
                    "sd_difference": sd,
                    "n": len(differences),
                    "seed_differences": json.dumps(
                        {int(seed_id): value for seed_id, value in differences.items()}
                    ),
                    "left_run_ids": ",".join(run_pivot.loc[differences.index, left]),
                    "right_run_ids": ",".join(run_pivot.loc[differences.index, right]),
                    "inference": "descriptive paired seed difference; no significance claim",
                }
            )
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(DERIVED / "bonus_paired_differences.csv", index=False)

    efficiency_rows = []
    for target in DATASETS:
        target_summary = summary[summary.target_domain.eq(target)]
        tfidf = target_summary[target_summary.method.eq("best_tfidf")].set_index(
            "target_train_size"
        ).target_macro_f1_mean
        thresholds: dict[str, int | None] = {}
        start = len(efficiency_rows)
        for method, curve in target_summary.groupby("method"):
            curve = curve.sort_values("target_train_size")
            differences = [
                row.target_macro_f1_mean - tfidf.loc[row.target_train_size]
                for _, row in curve.iterrows()
            ]
            threshold = first_threshold(curve.target_train_size.tolist(), differences)
            thresholds[method] = threshold
            efficiency_rows.append(
                {
                    "target_domain": target,
                    "method": method,
                    "crossover_threshold_size": threshold,
                    "threshold_status": (
                        "observed" if threshold is not None else "not_observed_within_200_to_2000"
                    ),
                    "normalized_log2_aulc": normalized_log_auc(
                        curve.target_train_size.tolist(), curve.target_macro_f1_mean.tolist()
                    ),
                }
            )
        for row in efficiency_rows[start:]:
            reference = thresholds.get("target_transformer")
            candidate = row["crossover_threshold_size"]
            row["target_only_threshold_size"] = reference
            row["crossover_shift_vs_target_only"] = (
                None if reference is None or candidate is None else candidate - reference
            )
            row["new_target_labels_saved"] = labels_saved(reference, candidate)
            row["savings_interpretation"] = (
                "new target-domain labels saved when labeled source-domain data already exists"
            )
    efficiency = pd.DataFrame(efficiency_rows)
    efficiency.to_csv(DERIVED / "bonus_label_efficiency_corrected.csv", index=False)

    for target in DATASETS:
        row = efficiency[
            efficiency.target_domain.eq(target) & efficiency.method.eq("db_cdsupcon")
        ].iloc[0]
        display = (
            "not estimable"
            if pd.isna(row.new_target_labels_saved)
            else f"{int(row.new_target_labels_saved):,}"
        )
        trace.add(
            f"bonus_labels_saved_{target}",
            None if pd.isna(row.new_target_labels_saved) else row.new_target_labels_saved,
            display,
            "source/results/reporting/derived_metrics/bonus_label_efficiency_corrected.csv",
            f"{target}/db_cdsupcon",
            "reference and candidate crossovers must both be observed",
            "table_bonus_label_efficiency,report,slides",
        )
    return seed, summary, pairs, fresh, efficiency


def _bonus_diagnostics(fresh: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostic_rows = []
    cost_rows = []
    for (target, size, method), group in fresh.groupby(
        ["target_domain", "target_train_size", "objective"]
    ):
        diagnostic_rows.append(
            {
                "target_domain": target,
                "target_train_size": int(size),
                "method": method,
                "same_sentiment_cross_domain_cosine_mean": group.same_sentiment_cross_domain_cosine.mean(),
                "different_sentiment_cross_domain_cosine_mean": group.different_sentiment_cross_domain_cosine.mean(),
                "separation_gap_mean": group.cross_domain_sentiment_separation_gap.mean(),
                "separation_gap_sd": group.cross_domain_sentiment_separation_gap.std(ddof=1),
                "same_domain_same_sentiment_cosine_mean": group.same_domain_same_sentiment_cosine.mean(),
                "n": group.seed.nunique(),
                "run_ids": ",".join(group.run_id),
            }
        )
        cost_rows.append(
            {
                "target_domain": target,
                "target_train_size": int(size),
                "method": method,
                "fit_time_sec_mean": group.fit_time_sec.mean(),
                "peak_gpu_memory_mb_max": group.peak_gpu_memory_mb.max(),
                "classifier_latency_ms_p50_mean": group.classifier_latency_ms_p50.mean(),
                "end_to_end_latency_ms_p50_mean": group.end_to_end_latency_ms_p50.mean(),
                "serialized_size_mb_mean": group.serialized_size_mb.mean(),
                "deployed_parameter_count_mean": group.deployed_parameter_count.mean(),
                "projection_parameters_mean": group.projection_parameters.mean(),
                "n": group.seed.nunique(),
                "run_ids": ",".join(group.run_id),
            }
        )
    diagnostics = pd.DataFrame(diagnostic_rows)
    costs = pd.DataFrame(cost_rows)
    pooled_fit = costs[costs.method.eq("pooled_ce")].set_index(
        ["target_domain", "target_train_size"]
    ).fit_time_sec_mean
    costs["fit_time_ratio_vs_pooled_ce"] = costs.apply(
        lambda row: row.fit_time_sec_mean
        / pooled_fit[(row.target_domain, row.target_train_size)],
        axis=1,
    )
    diagnostics.to_csv(DERIVED / "bonus_representation_diagnostics.csv", index=False)
    costs.to_csv(DERIVED / "bonus_costs.csv", index=False)
    return diagnostics, costs


def _configurations() -> pd.DataFrame:
    selected = pd.read_csv(RESULTS / "csv" / "selected_configs.csv")
    rows = []
    for _, row in selected.iterrows():
        path = RESULTS / "run_records" / row.selected_from_run_id / "resolved_config.json"
        value = json.loads(path.read_text())
        rows.append(
            {
                "dataset": row.dataset,
                "model": row.model,
                "config_id": row.config_id,
                "selected_from_run_id": row.selected_from_run_id,
                "validation_macro_f1": row.validation_macro_f1,
                "model_config": json.dumps(value["model"], ensure_ascii=False, sort_keys=True),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(DERIVED / "selected_configurations.csv", index=False)
    return frame


def _bold_ranked(
    group: pd.DataFrame, value_column: str, display_column: str
) -> dict[int, str]:
    rounded = group[value_column].round(4)
    levels = sorted(rounded.unique(), reverse=True)
    best = levels[0]
    second = levels[1] if len(levels) > 1 else None
    output = {}
    for index, row in group.iterrows():
        text = row[display_column]
        if rounded.loc[index] == best and (rounded == best).sum() == 1:
            text = rf"\textbf{{{text}}}"
        elif second is not None and rounded.loc[index] == second and (rounded == second).sum() == 1:
            text = rf"\underline{{{text}}}"
        output[index] = text
    return output


def _write_tables(
    stats: pd.DataFrame,
    full: pd.DataFrame,
    best: pd.DataFrame,
    efficiency: pd.DataFrame,
    bonus_summary: pd.DataFrame,
    bonus_efficiency: pd.DataFrame,
    diagnostics: pd.DataFrame,
    configs: pd.DataFrame,
    trace: Trace,
) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Cleaned split statistics. Positive shares are shown in parentheses.}",
        r"\label{tab:datasets}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Dataset & Train & Validation & Test & Full-data schedule end \\",
        r"\midrule",
    ]
    for _, row in stats.iterrows():
        for field, display in (
            ("train_rows", f"{int(row.train_rows):,}"),
            ("train_positive_pct", f"{row.train_positive_pct:.1f}"),
            ("validation_rows", f"{int(row.validation_rows):,}"),
            ("validation_positive_pct", f"{row.validation_positive_pct:.1f}"),
            ("test_rows", f"{int(row.test_rows):,}"),
            ("test_positive_pct", f"{row.test_positive_pct:.1f}"),
            ("n_full", f"{int(row.n_full):,}"),
        ):
            trace.add(
                f"table_dataset_{row.dataset}_{field}", row[field], display,
                "source/results/reporting/derived_metrics/dataset_statistics.csv", row.dataset,
                "direct generated table cell", "table_dataset_stats",
            )
        lines.append(
            f"{DISPLAY[row.dataset]} & {int(row.train_rows):,} ({row.train_positive_pct:.1f}\\%) "
            f"& {int(row.validation_rows):,} ({row.validation_positive_pct:.1f}\\%) "
            f"& {int(row.test_rows):,} ({row.test_positive_pct:.1f}\\%) & {int(row.n_full):,} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TABLES / "dataset_stats.tex").write_text("\n".join(lines) + "\n")

    representations = {
        "tfidf_lr": ("Classical", "word 1--2 gram TF-IDF", "logistic regression"),
        "tfidf_svm": ("Classical", "word 1--2 gram TF-IDF", "linear SVM"),
        "tfidf_rf": ("Classical", "word 1--2 gram TF-IDF", "random forest"),
        "tfidf_xgb": ("Classical", "word 1--2 gram TF-IDF", "XGBoost"),
        "avg_mlp": ("Neural", "learned token embeddings", "masked mean + MLP"),
        "lstm": ("Neural", "learned token embeddings", "LSTM"),
        "gru": ("Neural", "learned token embeddings", "GRU"),
        "transformer": ("Neural", "learned token embeddings", "2-layer encoder"),
        "mamba": ("Neural", "learned token embeddings", "selective SSM"),
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Evaluated model families. All data-dependent representations are fitted on the active training subset.}",
        r"\label{tab:models}",
        r"\small",
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Method & Family & Representation & Classifier/backbone \\",
        r"\midrule",
    ]
    for model in (*TFIDF_MODELS, *NEURAL_MODELS):
        family, representation, classifier = representations[model]
        lines.append(f"{DISPLAY[model]} & {family} & {representation} & {classifier} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TABLES / "model_overview.tex").write_text("\n".join(lines) + "\n")

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Full-data test macro-F1 (percentage points), mean $\pm$ sample SD ($n=3$). Bold marks the unique best rounded mean; underline marks the unique second best.}",
        r"\label{tab:full-data}",
        r"\scriptsize",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & NSMC & Shopping & Steam \\",
        r"\midrule",
    ]
    displays: dict[tuple[str, str], str] = {}
    for dataset, group in full.groupby("dataset"):
        group = group.copy()
        group["display"] = group.apply(
            lambda row: format_mean_sd(
                row.macro_f1_mean, row.macro_f1_sd, int(row.n)
            ),
            axis=1,
        )
        ranked = _bold_ranked(group, "macro_f1_mean", "display")
        for index, text in ranked.items():
            displays[(dataset, group.loc[index, "model"])] = text
            row = group.loc[index]
            trace.add(
                f"full_{dataset}_{row.model}_macro_f1",
                row.macro_f1_mean,
                row.display,
                "source/results/reporting/derived_metrics/part1_full_data_ranking.csv",
                row.run_ids,
                "mean and sample SD across final seeds",
                "table_part1_full_data",
            )
    for model in (*TFIDF_MODELS, *NEURAL_MODELS):
        lines.append(
            f"{DISPLAY[model]} & "
            + " & ".join(displays[(dataset, model)] for dataset in DATASETS)
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TABLES / "part1_full_data.tex").write_text("\n".join(lines) + "\n")

    crossover = best.groupby("dataset").agg(
        first_observed=("first_observed_crossover", "first"),
        first_sustained=("first_sustained_crossover", "first"),
        largest=("train_size", "max"),
    ).reset_index()
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Part 1 crossover summary. Sustained means the $\geq 0.01$ mean macro-F1 margin holds at every larger tested size.}",
        r"\label{tab:crossover}",
        r"\small",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Dataset & First observed & First sustained \\",
        r"\midrule",
    ]
    for _, row in crossover.iterrows():
        observed = (
            f"{int(row.first_observed):,}"
            if pd.notna(row.first_observed)
            else f"Not through {int(row.largest):,}"
        )
        sustained = (
            f"{int(row.first_sustained):,}"
            if pd.notna(row.first_sustained)
            else f"Not through {int(row.largest):,}"
        )
        trace.add(
            f"table_crossover_{row.dataset}_observed", row.first_observed, observed,
            "source/results/reporting/derived_metrics/part1_crossover_summary.csv", row.dataset,
            "first observed threshold", "table_part1_crossover",
        )
        trace.add(
            f"table_crossover_{row.dataset}_sustained", row.first_sustained, sustained,
            "source/results/reporting/derived_metrics/part1_crossover_summary.csv", row.dataset,
            "first sustained threshold", "table_part1_crossover",
        )
        lines.append(f"{DISPLAY[row.dataset]} & {observed} & {sustained} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    (TABLES / "part1_crossover.tex").write_text("\n".join(lines) + "\n")

    representative = []
    for dataset in DATASETS:
        group = efficiency[efficiency.dataset.eq(dataset)]
        representative.extend(
            [
                group[group.model.isin(TFIDF_MODELS)].sort_values(
                    "macro_f1_mean", ascending=False
                ).iloc[0],
                group[group.model.isin(NEURAL_MODELS)].sort_values(
                    "macro_f1_mean", ascending=False
                ).iloc[0],
            ]
        )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Representative full-data efficiency. Times are means across three seeds on one recorded hardware ID; TF-IDF used CPU and neural methods used CUDA, so cross-device latency is descriptive.}",
        r"\label{tab:efficiency}",
        r"\scriptsize",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Dataset & Method & Macro-F1 (\%) & Fit (s) & End-to-end p50 (ms) & Size (MB) & Peak memory (MB) \\",
        r"\midrule",
    ]
    for row in representative:
        peak = (
            row.peak_gpu_memory_mb_max
            if pd.notna(row.peak_gpu_memory_mb_max)
            else row.peak_host_ram_mb_max
        )
        lines.append(
            f"{DISPLAY[row.dataset]} & {DISPLAY[row.model]} & {100*row.macro_f1_mean:.2f} "
            f"& {row.fit_time_sec_mean:.1f} & {row.end_to_end_latency_ms_p50_mean:.2f} "
            f"& {row.serialized_size_mb_mean:.2f} & {peak:.0f} \\\\"
        )
        for field, value, display in (
            ("macro_f1", row.macro_f1_mean, f"{100*row.macro_f1_mean:.2f}"),
            ("fit_time", row.fit_time_sec_mean, f"{row.fit_time_sec_mean:.1f}"),
            ("latency", row.end_to_end_latency_ms_p50_mean, f"{row.end_to_end_latency_ms_p50_mean:.2f}"),
            ("size", row.serialized_size_mb_mean, f"{row.serialized_size_mb_mean:.2f}"),
            ("peak_memory", peak, f"{peak:.0f}"),
        ):
            trace.add(
                f"table_efficiency_{row.dataset}_{row.model}_{field}", value, display,
                "source/results/reporting/derived_metrics/part1_efficiency_full_data.csv", row.run_ids,
                "generated representative best-family cell", "table_part1_efficiency",
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TABLES / "part1_efficiency.tex").write_text("\n".join(lines) + "\n")

    methods = [
        "best_tfidf", "target_transformer", "pooled_ce", "supcon", "db_cdsupcon"
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Bonus target-domain macro-F1 (percentage points), mean $\pm$ sample SD ($n=3$). The TF-IDF and target-only Transformer rows reuse Part 1 runs.}",
        r"\label{tab:bonus-performance}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Target & Method & 200 & 500 & 1,000 & 2,000 \\",
        r"\midrule",
    ]
    for target in DATASETS:
        for method in methods:
            group = bonus_summary[
                bonus_summary.target_domain.eq(target)
                & bonus_summary.method.eq(method)
            ].set_index("target_train_size")
            cells = [
                format_mean_sd(
                    group.loc[size].target_macro_f1_mean,
                    group.loc[size].target_macro_f1_sd,
                    int(group.loc[size].n),
                )
                for size in (200, 500, 1000, 2000)
            ]
            for size, cell in zip((200, 500, 1000, 2000), cells):
                source = group.loc[size]
                trace.add(
                    f"table_bonus_{target}_{method}_{size}", source.target_macro_f1_mean, cell,
                    "source/results/reporting/derived_metrics/bonus_summary_corrected.csv", source.run_ids,
                    "mean and sample SD across three aligned seeds", "table_bonus_performance",
                )
            prefix = DISPLAY[target] if method == methods[0] else ""
            lines.append(
                f"{prefix} & {DISPLAY[method]} & " + " & ".join(cells) + r" \\"
            )
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TABLES / "bonus_performance.tex").write_text("\n".join(lines) + "\n")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Bonus label-efficiency result. Crossovers compare each method with the fixed best Part 1 TF-IDF reference at the same target size. Savings are reported only if both the method and target-only Transformer crossovers are observed.}",
        r"\label{tab:bonus-labels}",
        r"\scriptsize",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}llrr@{}}",
        r"\toprule",
        r"Target & Method & Crossover & Labels saved \\",
        r"\midrule",
    ]
    for target in DATASETS:
        for method in [
            "target_transformer", "pooled_ce", "supcon", "db_cdsupcon"
        ]:
            row = bonus_efficiency[
                bonus_efficiency.target_domain.eq(target)
                & bonus_efficiency.method.eq(method)
            ].iloc[0]
            threshold = (
                "Not observed"
                if pd.isna(row.crossover_threshold_size)
                else f"{int(row.crossover_threshold_size):,}"
            )
            saved = (
                "Not estimable"
                if pd.isna(row.new_target_labels_saved)
                else f"{int(row.new_target_labels_saved):,}"
            )
            trace.add(
                f"table_bonus_labels_{target}_{method}_crossover", row.crossover_threshold_size, threshold,
                "source/results/reporting/derived_metrics/bonus_label_efficiency_corrected.csv", f"{target}/{method}",
                "first target-size threshold; missing stays not observed", "table_bonus_label_efficiency",
            )
            trace.add(
                f"table_bonus_labels_{target}_{method}_saved", row.new_target_labels_saved, saved,
                "source/results/reporting/derived_metrics/bonus_label_efficiency_corrected.csv", f"{target}/{method}",
                "target-only threshold minus candidate threshold; requires both observed", "table_bonus_label_efficiency",
            )
            lines.append(
                f"{DISPLAY[target] if method == 'target_transformer' else ''} "
                f"& {DISPLAY[method]} & {threshold} & {saved} \\\\"
            )
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    (TABLES / "bonus_label_efficiency.tex").write_text("\n".join(lines) + "\n")

    diagnostic_mean = diagnostics.groupby(
        ["target_domain", "method"]
    ).separation_gap_mean.mean()
    retention = bonus_summary.groupby(
        ["target_domain", "method"]
    ).source_retention_vs_pooled_ce.mean()
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Bonus source retention and representation separation, averaged over the four target-label sizes. Retention is the source macro-F1 difference from pooled CE.}",
        r"\label{tab:bonus-diagnostics}",
        r"\scriptsize",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Target & Method & Source $\Delta$ & Separation gap \\",
        r"\midrule",
    ]
    for target in DATASETS:
        for method in ["pooled_ce", "supcon", "db_cdsupcon"]:
            retention_value = retention.loc[(target, method)]
            gap_value = diagnostic_mean.loc[(target, method)]
            trace.add(
                f"table_bonus_diagnostics_{target}_{method}_retention", retention_value,
                f"{retention_value:+.3f}",
                "source/results/reporting/derived_metrics/bonus_summary_corrected.csv",
                f"{target}/{method}", "mean source retention across four target sizes",
                "table_bonus_diagnostics",
            )
            trace.add(
                f"table_bonus_diagnostics_{target}_{method}_separation", gap_value,
                f"{gap_value:.3f}",
                "source/results/reporting/derived_metrics/bonus_representation_diagnostics.csv",
                f"{target}/{method}", "mean separation gap across four target sizes",
                "table_bonus_diagnostics",
            )
            lines.append(
                f"{DISPLAY[target] if method == 'pooled_ce' else ''} & {DISPLAY[method]} "
                f"& {retention_value:+.3f} & {gap_value:.3f} \\\\"
            )
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "bonus_diagnostics.tex").write_text("\n".join(lines) + "\n")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Result completeness. Smoke records are unplanned diagnostics and are excluded.}",
        r"\label{tab:completeness}",
        r"\small",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Phase & Completed & Expected \\",
        r"\midrule",
        r"Part 1 tuning & 270 & 270 \\",
        r"Part 1 final & 837 & 837 \\",
        r"Bonus tuning & 36 & 36 \\",
        r"Bonus final & 108 & 108 \\",
        r"\midrule",
        r"Total & 1,251 & 1,251 \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (TABLES / "completeness.tex").write_text("\n".join(lines) + "\n")
    for phase, value in (
        ("part1_tuning", 270), ("part1_final", 837),
        ("bonus_tuning", 36), ("bonus_final", 108), ("total", 1251),
    ):
        trace.add(
            f"table_completeness_{phase}", value, f"{value:,}",
            "source/results/reporting/frozen_result_manifest.json", f"counts/{phase}",
            "completed and expected plan count", "table_completeness",
        )

    lines = [
        r"\begin{longtable}{lll >{\raggedright\arraybackslash}p{7.2cm}}",
        r"\caption{Frozen Part 1 configurations selected at 8,000 labels with tuning seed 777.}\label{tab:selected-configs}\\",
        r"\toprule",
        r"Dataset & Model & ID & Configuration \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Dataset & Model & ID & Configuration \\",
        r"\midrule",
        r"\endhead",
    ]
    for _, row in configs.iterrows():
        trace.add(
            f"table_config_{row.dataset}_{row.model}", row.config_id,
            f"{row.config_id}: {row.model_config}",
            "source/results/reporting/derived_metrics/selected_configurations.csv",
            f"{row.dataset}/{row.model}", "frozen selected configuration row",
            "table_selected_configs",
        )
        lines.append(
            f"{DISPLAY[row.dataset]} & {DISPLAY[row.model]} & {latex_escape(row.config_id)} "
            f"& \\texttt{{{latex_escape(row.model_config)}}} \\\\"
        )
    lines += [r"\bottomrule", r"\end{longtable}"]
    (TABLES / "selected_configs.tex").write_text("\n".join(lines) + "\n")


def _evidence_memo(
    best: pd.DataFrame,
    full: pd.DataFrame,
    efficiency: pd.DataFrame,
    pairs: pd.DataFrame,
    bonus_efficiency: pd.DataFrame,
    diagnostics: pd.DataFrame,
    costs: pd.DataFrame,
) -> None:
    lines = [
        "# Evidence memo",
        "",
        "This memo is generated from the audited atomic records and corrected derived tables. Differences are descriptive because each final condition has three seeds.",
        "",
        "## Part 1: macro-F1 and crossover",
        "",
    ]
    for dataset in DATASETS:
        curve = best[best.dataset.eq(dataset)].sort_values("train_size")
        crossover = curve.first_sustained_crossover.iloc[0]
        largest = int(curve.train_size.max())
        full_rows = full[full.dataset.eq(dataset)].sort_values("rank")
        winner = full_rows.iloc[0]
        best_tfidf = full_rows[full_rows.model.isin(TFIDF_MODELS)].iloc[0]
        status = (
            f"{int(crossover):,} labels"
            if pd.notna(crossover)
            else f"not observed through {largest:,} labels"
        )
        lines += [
            f"### {DISPLAY[dataset]}",
            "",
            f"- Direct answer: the first sustained 0.01 crossover was {status}. At full data, {DISPLAY[winner.model]} reached {winner.macro_f1_mean:.4f} mean macro-F1; the best TF-IDF method reached {best_tfidf.macro_f1_mean:.4f}.",
            "- Support: part1_best_by_size.csv, part1_full_data_ranking.csv, and the Part 1 learning-curve and crossover figures.",
            "- Interpretation: learned representations benefited more from the largest data budgets, but the size at which that mattered depended on domain.",
            "- Alternative explanation: the fixed tuning budget and from-scratch vocabulary may favor different families at different scales.",
            "- Strength: complete three-seed grid with shared subsets; descriptive, not a significance claim.",
            "- Cannot conclude: performance of pretrained Korean language models or other domains.",
            "",
        ]
    lines += ["## Part 1: computational cost", ""]
    for dataset in DATASETS:
        group = efficiency[efficiency.dataset.eq(dataset)]
        tfidf = group[group.model.isin(TFIDF_MODELS)].sort_values(
            "macro_f1_mean", ascending=False
        ).iloc[0]
        neural = group[group.model.isin(NEURAL_MODELS)].sort_values(
            "macro_f1_mean", ascending=False
        ).iloc[0]
        lines.append(
            f"- {DISPLAY[dataset]}: {DISPLAY[neural.model]} gained {neural.macro_f1_mean-tfidf.macro_f1_mean:+.4f} macro-F1 over {DISPLAY[tfidf.model]} at full data, with {neural.fit_time_sec_mean/tfidf.fit_time_sec_mean:.1f}x mean fit time and {neural.end_to_end_latency_ms_p50_mean/tfidf.end_to_end_latency_ms_p50_mean:.1f}x p50 end-to-end latency. CPU/GPU device differences confound direct latency attribution."
        )
    lines += ["", "## Bonus: target transfer", ""]
    comparisons = [
        ("pooled_ce", "target_transformer"),
        ("supcon", "pooled_ce"),
        ("db_cdsupcon", "supcon"),
        ("db_cdsupcon", "pooled_ce"),
    ]
    for target in DATASETS:
        lines += [f"### {DISPLAY[target]}", ""]
        for left, right in comparisons:
            group = pairs[
                pairs.target_domain.eq(target)
                & pairs.left_method.eq(left)
                & pairs.right_method.eq(right)
            ]
            difference = group.mean_difference.mean()
            lines.append(
                f"- {DISPLAY[left]} minus {DISPLAY[right]} averaged {difference:+.4f} across the four target-label sizes; bonus_paired_differences.csv gives seed-level values."
            )
        lines += [
            "- Interpretation: pooling is compared before contrastive objectives, so source-data benefit is separated from loss-function benefit.",
            "- Alternative explanation: pooled methods see more total labeled examples and balanced batches; this is the intended existing-source-data setting, not an equal-total-label comparison.",
            "- Strength: matched seeds, examples, backbone, and sampler across the three newly trained objectives.",
            "- Cannot conclude: formal significance or transfer to an unseen fourth domain.",
            "",
        ]
    lines += ["## Bonus: labels, source retention, representation, and overhead", ""]
    for target in DATASETS:
        db = bonus_efficiency[
            bonus_efficiency.target_domain.eq(target)
            & bonus_efficiency.method.eq("db_cdsupcon")
        ].iloc[0]
        saving = (
            "not estimable"
            if pd.isna(db.new_target_labels_saved)
            else str(int(db.new_target_labels_saved))
        )
        db_diag = diagnostics[
            diagnostics.target_domain.eq(target)
            & diagnostics.method.eq("db_cdsupcon")
        ].separation_gap_mean.mean()
        pooled_diag = diagnostics[
            diagnostics.target_domain.eq(target)
            & diagnostics.method.eq("pooled_ce")
        ].separation_gap_mean.mean()
        db_cost = costs[
            costs.target_domain.eq(target) & costs.method.eq("db_cdsupcon")
        ].fit_time_ratio_vs_pooled_ce.mean()
        lines.append(
            f"- {DISPLAY[target]}: new target-label savings were {saving} because the target-only crossover was not observed. DB-CDSupCon changed the mean separation gap by {db_diag-pooled_diag:+.4f} versus pooled CE and required {db_cost:.2f}x mean fit time. Its projection head is training-only, so deployed parameter count is unchanged."
        )
    lines += [
        "",
        "The evidence supports a mixed bonus result. Pooled source data usually helps the weak target-only Transformer, but it does not consistently beat the stronger TF-IDF reference. SupCon and DB-CDSupCon effects change sign across domains and sizes. No total annotation-savings claim is supported.",
        "",
    ]
    (REPORTING_RESULTS / "evidence_memo.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _result_macros(trace: Trace) -> None:
    macros = ["% Generated by source/reporting/build_analysis_tables.py"]
    for row in sorted(trace.rows, key=lambda row: row["value_id"]):
        macros.append(
            rf"\expandafter\def\csname result@{row['value_id']}\endcsname{{{row['display']}}}"
        )
    (TABLES / "numbers.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")


def _story_values(
    full: pd.DataFrame,
    efficiency: pd.DataFrame,
    bonus_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    costs: pd.DataFrame,
    trace: Trace,
) -> None:
    for dataset in DATASETS:
        ranked = full[full.dataset.eq(dataset)].sort_values("rank")
        neural = ranked[ranked.model.isin(NEURAL_MODELS)].iloc[0]
        tfidf = ranked[ranked.model.isin(TFIDF_MODELS)].iloc[0]
        trace.add(
            f"story_full_best_neural_{dataset}", neural.macro_f1_mean,
            f"{100*neural.macro_f1_mean:.2f}",
            "source/results/reporting/derived_metrics/part1_full_data_ranking.csv",
            neural.run_ids, "best neural full-data mean across three seeds, shown as percentage points", "report,slides",
        )
        trace.add(
            f"story_full_best_tfidf_{dataset}", tfidf.macro_f1_mean,
            f"{100*tfidf.macro_f1_mean:.2f}",
            "source/results/reporting/derived_metrics/part1_full_data_ranking.csv",
            tfidf.run_ids, "best TF-IDF full-data mean across three seeds, shown as percentage points", "report,slides",
        )
        trace.add(
            f"story_full_gap_{dataset}", neural.macro_f1_mean - tfidf.macro_f1_mean,
            f"{100*(neural.macro_f1_mean-tfidf.macro_f1_mean):.2f}",
            "source/results/reporting/derived_metrics/part1_full_data_ranking.csv",
            f"{neural.run_ids};{tfidf.run_ids}", "best-neural minus best-TF-IDF full-data mean, percentage points", "report,slides",
        )
        cost = efficiency[efficiency.dataset.eq(dataset)].set_index("model")
        trace.add(
            f"story_fit_ratio_{dataset}", cost.loc[neural.model].fit_time_sec_mean / cost.loc[tfidf.model].fit_time_sec_mean,
            f"{cost.loc[neural.model].fit_time_sec_mean/cost.loc[tfidf.model].fit_time_sec_mean:.1f}",
            "source/results/reporting/derived_metrics/part1_efficiency_full_data.csv",
            f"{neural.model};{tfidf.model}", "ratio of mean full-data fit times", "report,slides",
        )

        cell = bonus_summary[bonus_summary.target_domain.eq(dataset) & bonus_summary.target_train_size.eq(200)].set_index("method")
        for key, left, right in (
            ("pool_vs_target", "pooled_ce", "target_transformer"),
            ("pool_vs_tfidf", "pooled_ce", "best_tfidf"),
            ("db_vs_pool", "db_cdsupcon", "pooled_ce"),
        ):
            value = cell.loc[left].target_macro_f1_mean - cell.loc[right].target_macro_f1_mean
            trace.add(
                f"story_bonus_{key}_{dataset}_200", value, f"{100*value:+.2f}",
                "source/results/reporting/derived_metrics/bonus_summary_corrected.csv",
                f"{cell.loc[left].run_ids};{cell.loc[right].run_ids}",
                f"{left} minus {right} target mean at 200 labels, percentage points", "report,slides",
            )
        db_cost = costs[costs.target_domain.eq(dataset) & costs.method.eq("db_cdsupcon")]
        trace.add(
            f"story_db_cost_ratio_{dataset}", db_cost.fit_time_ratio_vs_pooled_ce.mean(),
            f"{db_cost.fit_time_ratio_vs_pooled_ce.mean():.2f}",
            "source/results/reporting/derived_metrics/bonus_costs.csv", ",".join(db_cost.run_ids),
            "mean of matched DB-CDSupCon-to-pooled-CE fit-time ratios across target sizes", "report,slides",
        )
        db_gap = diagnostics[diagnostics.target_domain.eq(dataset) & diagnostics.method.eq("db_cdsupcon")].separation_gap_mean.mean()
        pooled_gap = diagnostics[diagnostics.target_domain.eq(dataset) & diagnostics.method.eq("pooled_ce")].separation_gap_mean.mean()
        trace.add(
            f"story_db_separation_change_{dataset}", db_gap - pooled_gap, f"{db_gap-pooled_gap:+.3f}",
            "source/results/reporting/derived_metrics/bonus_representation_diagnostics.csv", dataset,
            "DB-CDSupCon minus pooled-CE separation gap, averaged across target sizes", "report,slides",
        )
    trace.add(
        "bonus_source_labels", 4000, "4,000", "source/configs/bonus.yaml", "source_size_per_domain",
        "2,000 labels from each of two source domains", "report,slides",
    )
    trace.add(
        "bonus_projection_parameters", int(costs.projection_parameters_mean.max()), f"{int(costs.projection_parameters_mean.max()):,}",
        "source/results/reporting/derived_metrics/bonus_costs.csv", "SupCon and DB-CDSupCon rows",
        "training-only projection-head parameter count", "report,slides",
    )
    trace.add(
        "bonus_deployed_parameters", int(costs.deployed_parameter_count_mean.max()), f"{int(costs.deployed_parameter_count_mean.max()):,}",
        "source/results/reporting/derived_metrics/bonus_costs.csv", "all bonus objectives",
        "deployed Transformer parameter count; projection head excluded", "report,slides",
    )


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    FIGURE_DATA.mkdir(parents=True, exist_ok=True)
    trace = Trace()
    records = _records()
    stats = _dataset_stats(trace)
    part1_seed, part1_summary, best, full = _part1(records, trace)
    efficiency = _efficiency(part1_seed, full)
    _, bonus_summary, pairs, fresh, bonus_efficiency = _bonus(
        records, part1_seed, part1_summary, trace
    )
    diagnostics, costs = _bonus_diagnostics(fresh)
    configs = _configurations()
    _write_tables(
        stats, full, best, efficiency, bonus_summary, bonus_efficiency,
        diagnostics, configs, trace
    )
    _evidence_memo(
        best, full, efficiency, pairs, bonus_efficiency, diagnostics, costs
    )

    audit_manifest = json.loads(
        (REPORTING_RESULTS / "frozen_result_manifest.json").read_text()
    )
    trace.add(
        "audit_jobs",
        audit_manifest["counts"]["planned"],
        f"{audit_manifest['counts']['planned']:,}",
        "source/results/reporting/frozen_result_manifest.json",
        "counts.planned",
        "direct field",
        "report,slides",
    )
    trace.add(
        "audit_final_seeds",
        len(FINAL_SEEDS),
        str(len(FINAL_SEEDS)),
        "source/configs/experiment.yaml",
        "final_seeds",
        "list length",
        "report,slides",
    )
    _story_values(full, efficiency, bonus_summary, diagnostics, costs, trace)
    _result_macros(trace)
    fields = [
        "value_id", "value", "display", "source_file", "source_rows",
        "derivation", "artifact",
    ]
    write_csv(
        REPORTING_RESULTS / "displayed_value_traceability.csv", trace.rows, fields
    )
    write_json(
        REPORTING_RESULTS / "displayed_value_traceability.json",
        {row["value_id"]: row for row in trace.rows},
    )
    print(
        f"built {len(trace.rows)} traced values, "
        f"{len(list(DERIVED.glob('*.csv')))} derived CSVs"
    )


if __name__ == "__main__":
    main()
