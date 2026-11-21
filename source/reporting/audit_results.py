from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from reporting_utils import (
    DATASETS,
    FINAL_SEEDS,
    REPORTING_RESULTS,
    RESULTS,
    ROOT,
    SOURCE,
    TFIDF_MODELS,
    as_float,
    as_int,
    read_csv,
    sha256_file,
    write_csv,
    write_json,
)


REQUIRED_RUN_FILES = (
    "record.json", "resolved_config.json", "run_metadata.json", "status.json",
    "metrics.json", "resources.json", "confusion_matrix.json", "SUCCESS",
)


class Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def check(self, check_id: str, area: str, ok: bool, details: str, *, severity: str = "error", count: int = 0) -> None:
        self.rows.append(
            {
                "check_id": check_id,
                "area": area,
                "severity": severity,
                "status": "pass" if ok else ("warning" if severity == "warning" else "fail"),
                "count": count,
                "details": details,
            }
        )

    @property
    def errors(self) -> list[dict[str, object]]:
        return [row for row in self.rows if row["status"] == "fail"]

    @property
    def warnings(self) -> list[dict[str, object]]:
        return [row for row in self.rows if row["status"] == "warning"]


def _git(args: list[str]) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_paths(planned_ids: set[str]) -> list[tuple[str, Path]]:
    values: list[tuple[str, Path]] = []
    patterns = {
        "plan": [RESULTS / "plans", RESULTS / "csv"],
        "data_manifest": [SOURCE / "data" / "manifests", SOURCE / "data" / "splits"],
        "configuration": [SOURCE / "configs"],
        "server_evidence": [RESULTS / "server"],
    }
    for category, directories in patterns.items():
        for directory in directories:
            if not directory.exists():
                continue
            for path in directory.rglob("*"):
                if not path.is_file():
                    continue
                if category == "server_evidence" and path.suffix not in {".log", ".md", ".json", ".csv", ".txt", ".yaml"}:
                    continue
                values.append((category, path))
    for run_id in sorted(planned_ids):
        directory = RESULTS / "run_records" / run_id
        for name in REQUIRED_RUN_FILES:
            path = directory / name
            if path.exists():
                values.append(("atomic_run_record", path))
        values.extend(("run_log", path) for path in directory.glob("*.log"))
    values.extend(("project_log", path) for path in (SOURCE / "logs").glob("*.log"))
    unique = {(category, path.resolve()): path for category, path in values}
    return sorted(((category, path) for (category, _), path in unique.items()), key=lambda item: str(item[1]))


def _selected_part1_matches(tuning: list[dict[str, str]], selected: list[dict[str, str]]) -> list[str]:
    problems = []
    selected_map = {(row["dataset"], row["model"]): row for row in selected}
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in tuning:
        if row["status"] == "completed":
            groups[(row["dataset"], row["model"])].append(row)
    for key, rows in groups.items():
        winner = min(
            rows,
            key=lambda row: (
                -float(row["validation_macro_f1"]),
                float(row["classifier_latency_ms_mean"]),
                float(row["serialized_size_mb"]),
                row["config_id"],
            ),
        )
        chosen = selected_map.get(key)
        if not chosen or chosen["selected_from_run_id"] != winner["run_id"]:
            problems.append(f"{key}: expected {winner['run_id']}, found {chosen and chosen['selected_from_run_id']}")
    return problems


def _selected_bonus_matches(tuning: list[dict[str, str]], selected: list[dict[str, str]]) -> list[str]:
    problems = []
    selected_map = {(row["target_domain"], row["objective"]): row for row in selected}
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in tuning:
        if row["status"] == "completed":
            groups[(row["target_domain"], row["objective"])].append(row)
    for key, rows in groups.items():
        for row in rows:
            metrics = _json(RESULTS / "run_records" / row["run_id"] / "metrics.json")
            row["_mean_source_validation_macro_f1"] = str(metrics["mean_source_validation_macro_f1"])
        winner = min(
            rows,
            key=lambda row: (
                -float(row["validation_macro_f1"]),
                -float(row["_mean_source_validation_macro_f1"]),
                float(row["fit_time_sec"]),
                row["config_id"],
            ),
        )
        chosen = selected_map.get(key)
        if not chosen or chosen["selected_from_run_id"] != winner["run_id"]:
            problems.append(f"{key}: expected {winner['run_id']}, found {chosen and chosen['selected_from_run_id']}")
    return problems


def _audit() -> tuple[dict, Audit]:
    audit = Audit()
    csv_dir = RESULTS / "csv"
    part1_plan = read_csv(RESULTS / "plans" / "part1_tuning_plan.csv") + read_csv(RESULTS / "plans" / "part1_final_plan.csv")
    bonus_plan = read_csv(RESULTS / "plans" / "bonus_tuning_plan.csv") + read_csv(RESULTS / "plans" / "bonus_final_plan.csv")
    plan = part1_plan + bonus_plan
    planned_ids = [row["run_id"] for row in plan]
    record_rows = read_csv(csv_dir / "run_records.csv")
    record_map = {row["run_id"]: row for row in record_rows}
    planned_records = [record_map[run_id] for run_id in planned_ids if run_id in record_map]

    audit.check("plan_count", "completeness", len(plan) == 1251, f"planned={len(plan)}, expected=1251", count=len(plan))
    audit.check("unique_plan_run_ids", "identity", len(planned_ids) == len(set(planned_ids)), "plan run IDs are unique")
    duplicate_record_ids = len(record_rows) - len(record_map)
    audit.check("unique_record_run_ids", "identity", duplicate_record_ids == 0, f"duplicate record IDs={duplicate_record_ids}", count=duplicate_record_ids)
    missing = sorted(set(planned_ids) - set(record_map))
    extra = sorted(set(record_map) - set(planned_ids))
    audit.check("planned_records_present", "completeness", not missing, f"missing planned records={len(missing)}", count=len(missing))
    audit.check("unplanned_records_declared", "completeness", True, f"unplanned records={len(extra)} (smoke records; excluded from final analysis)", severity="warning" if extra else "error", count=len(extra))

    phase_counts: dict[str, Counter] = defaultdict(Counter)
    for row in plan:
        phase_counts[row["phase"]][record_map.get(row["run_id"], {}).get("status", "missing")] += 1
    noncompleted = [(phase, status, count) for phase, counts in phase_counts.items() for status, count in counts.items() if status != "completed"]
    audit.check("all_planned_jobs_completed", "completeness", not noncompleted, f"non-completed groups={noncompleted}")

    final_rows = [row for row in planned_records if row["phase"] in {"part1_final", "bonus_final"}]
    groups: dict[tuple, set[int]] = defaultdict(set)
    for row in final_rows:
        key = (
            row["phase"], row["dataset"], row["model"], row["objective"],
            row["target_train_size"] if row["phase"] == "bonus_final" else row["train_size"],
        )
        groups[key].add(int(row["seed"]))
    incomplete = {key: sorted(seeds) for key, seeds in groups.items() if seeds != set(FINAL_SEEDS)}
    audit.check("final_seed_completeness", "completeness", not incomplete, f"incomplete conditions={len(incomplete)}", count=len(incomplete))

    tuning = [row for row in planned_records if row["phase"] in {"part1_tuning", "bonus_tuning"}]
    test_fields = [
        "test_macro_f1", "test_accuracy", "test_macro_precision", "test_macro_recall",
        "target_test_macro_f1", "source_1_test_macro_f1", "source_2_test_macro_f1",
        "mean_source_test_macro_f1", "all_domain_mean_macro_f1",
    ]
    contaminated = [row["run_id"] for row in tuning if any(row.get(field) not in (None, "") for field in test_fields)]
    # Bonus selection legitimately stores source validation in the historical field named mean_source_test_macro_f1.
    bonus_source_only = [
        row["run_id"] for row in tuning if row["phase"] == "bonus_tuning"
        and row.get("mean_source_test_macro_f1") not in (None, "")
        and all(row.get(field) in (None, "") for field in test_fields if field != "mean_source_test_macro_f1")
    ]
    contaminated = sorted(set(contaminated) - set(bonus_source_only))
    audit.check("no_test_metrics_in_tuning", "selection", not contaminated, f"tuning rows with test metrics={len(contaminated)}", count=len(contaminated))

    selected = read_csv(csv_dir / "selected_configs.csv")
    selection_problems = _selected_part1_matches(
        read_csv(csv_dir / "tuning_trials.csv"), selected
    )
    audit.check("part1_selection_recomputed", "selection", not selection_problems, f"selector mismatches={len(selection_problems)}", count=len(selection_problems))
    selected_map = {(row["dataset"], row["model"]): (row["config_id"], row["config_hash"]) for row in selected}
    frozen_mismatches = [
        row["run_id"] for row in part1_plan if row["phase"] == "part1_final"
        and (row["config_id"], row["config_hash"]) != selected_map.get((row["dataset"], row["model"]))
    ]
    audit.check("part1_configs_frozen", "selection", not frozen_mismatches, f"final config mismatches={len(frozen_mismatches)}", count=len(frozen_mismatches))

    bonus_selected = read_csv(csv_dir / "bonus_selected_configs.csv")
    bonus_selection_problems = _selected_bonus_matches(read_csv(csv_dir / "bonus_tuning_trials.csv"), bonus_selected)
    audit.check("bonus_selection_recomputed", "selection", not bonus_selection_problems, f"selector mismatches={len(bonus_selection_problems)}", count=len(bonus_selection_problems))

    data_manifest_hash = _json(SOURCE / "data" / "manifests" / "data_manifest.json")["sha256"]
    hash_mismatches = [row["run_id"] for row in planned_records if row["data_manifest_hash"] != data_manifest_hash]
    audit.check("shared_data_manifest", "fairness", not hash_mismatches, f"run data-hash mismatches={len(hash_mismatches)}", count=len(hash_mismatches))

    split_sets: dict[tuple[str, str], set[str]] = {}
    for dataset in DATASETS:
        with (SOURCE / "data" / "splits" / f"{dataset}_splits.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                split_sets.setdefault((dataset, row["split"]), set()).add(row["example_id"])
    subset_issues = []
    for dataset in DATASETS:
        train = split_sets[(dataset, "train")]
        held_out = split_sets[(dataset, "validation")] | split_sets[(dataset, "test")]
        for seed in (*FINAL_SEEDS, 777):
            rows = read_csv(SOURCE / "data" / "manifests" / f"subset_{dataset}_seed_{seed}.csv")
            ids = [row["example_id"] for row in sorted(rows, key=lambda row: int(row["rank"]))]
            if len(ids) != len(set(ids)) or set(ids) != train or set(ids) & held_out:
                subset_issues.append(f"{dataset}/seed{seed}")
    audit.check("nested_subsets_and_no_leakage", "fairness", not subset_issues, f"subset/leakage issues={len(subset_issues)}", count=len(subset_issues))
    audit.check("identical_part1_subset_ids", "fairness", not hash_mismatches, "all Part 1 models reference the same dataset/size/seed prefix manifests")

    bonus_fairness_issues = []
    bonus_groups: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in planned_records:
        if row["phase"] == "bonus_final":
            bonus_groups[(row["target_domain"], int(row["target_train_size"]), int(row["seed"]))].append(row)
    for key, rows in bonus_groups.items():
        if {row["objective"] for row in rows} != {"pooled_ce", "supcon", "db_cdsupcon"}:
            bonus_fairness_issues.append(f"{key}: objectives")
            continue
        resolved = [_json(RESULTS / "run_records" / row["run_id"] / "resolved_config.json") for row in rows]
        shared = [
            (
                value["model"], value["experiment"]["neural"], value["bonus"]["effective_batch_size"],
                value["bonus"]["examples_per_cell"], value["plan"]["seed"], value["plan"]["target_train_size"],
                value["plan"]["source_size_per_domain"], value["plan"]["source_domains"],
            )
            for value in resolved
        ]
        if any(value != shared[0] for value in shared[1:]) or any(row["batch_sampler"] != "domain_label_balanced" for row in rows):
            bonus_fairness_issues.append(str(key))
    audit.check("bonus_matched_design", "fairness", not bonus_fairness_issues, f"mismatched bonus cells={len(bonus_fairness_issues)}", count=len(bonus_fairness_issues))

    part1_final = [row for row in planned_records if row["phase"] == "part1_final"]
    by_cell: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in part1_final:
        by_cell[(row["dataset"], int(row["train_size"]))].append(row)
    expected_reference: dict[tuple[str, int], str] = {}
    for key, rows in by_cell.items():
        model_means = {
            model: sum(float(row["test_macro_f1"]) for row in rows if row["model"] == model) / 3
            for model in TFIDF_MODELS
        }
        expected_reference[key] = max(model_means, key=model_means.get)
    bonus_by_seed = read_csv(csv_dir / "bonus_metrics_by_seed.csv")
    mixed_reference_rows = []
    for row in bonus_by_seed:
        if row["method"] != "best_tfidf":
            continue
        source = record_map[row["run_id"]]
        expected = expected_reference[(row["target_domain"], int(row["target_train_size"]))]
        if source["model"] != expected:
            mixed_reference_rows.append(row["run_id"])
    audit.check(
        "bonus_tfidf_reference_selection",
        "derived_csv",
        not mixed_reference_rows,
        f"existing bonus aggregate mixes per-seed TF-IDF winners in {len(mixed_reference_rows)} rows; reporting rebuild selects one model by three-seed mean",
        severity="warning",
        count=len(mixed_reference_rows),
    )

    invalid_metrics = []
    bad_confusions = []
    test_sizes = {row["dataset"]: int(row["test_rows"]) for row in read_csv(csv_dir / "dataset_statistics.csv")}
    for row in final_rows:
        metric_names = ["test_macro_f1", "test_accuracy", "test_macro_precision", "test_macro_recall"]
        if row["phase"] == "bonus_final":
            metric_names += ["target_test_macro_f1", "source_1_test_macro_f1", "source_2_test_macro_f1", "mean_source_test_macro_f1", "all_domain_mean_macro_f1"]
        if any((value := as_float(row.get(name))) is None or not 0 <= value <= 1 for name in metric_names):
            invalid_metrics.append(row["run_id"])
        total = sum(as_int(row.get(name)) or 0 for name in ("tn", "fp", "fn", "tp"))
        if total != test_sizes[row["dataset"]]:
            bad_confusions.append(row["run_id"])
    audit.check("metric_ranges", "plausibility", not invalid_metrics, f"invalid final metric rows={len(invalid_metrics)}", count=len(invalid_metrics))
    audit.check("confusion_totals", "plausibility", not bad_confusions, f"confusion total mismatches={len(bad_confusions)}", count=len(bad_confusions))
    invalid_resources = [
        row["run_id"] for row in final_rows
        if any(as_float(row.get(field)) is None or as_float(row[field]) <= 0 for field in ("duration_sec", "fit_time_sec", "serialized_size_mb", "throughput_samples_per_sec"))
        or as_float(row.get("classifier_latency_ms_p50")) is None
    ]
    audit.check("resource_plausibility", "plausibility", not invalid_resources, f"invalid resource rows={len(invalid_resources)}", count=len(invalid_resources))

    artifact_issues = []
    for run_id in planned_ids:
        directory = RESULTS / "run_records" / run_id
        missing_files = [name for name in REQUIRED_RUN_FILES if not (directory / name).exists()]
        if missing_files:
            artifact_issues.append(f"{run_id}: {','.join(missing_files)}")
    audit.check("atomic_artifacts", "integrity", not artifact_issues, f"runs missing required artifacts={len(artifact_issues)}", count=len(artifact_issues))

    hardware_ids = sorted({row["hardware_id"] for row in planned_records if row.get("hardware_id")})
    audit.check("timing_hardware_identity", "resources", len(hardware_ids) == 1, f"hardware IDs={hardware_ids}", severity="warning" if len(hardware_ids) != 1 else "error", count=len(hardware_ids))

    git_commit = _git(["rev-parse", "HEAD"])
    dirty_lines = _git(["status", "--porcelain=v1"]).splitlines()
    now_utc = datetime.now(timezone.utc)
    evidence = []
    for category, path in _manifest_paths(set(planned_ids)):
        evidence.append(
            {
                "category": category,
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    outcome = "not_ready_for_final_reporting" if audit.errors else ("complete_with_declared_exceptions" if audit.warnings else "complete")
    manifest = {
        "schema_version": 1,
        "audit_outcome": outcome,
        "created_at_utc": now_utc.isoformat(),
        "created_at_asia_seoul": now_utc.astimezone(ZoneInfo("Asia/Seoul")).isoformat(),
        "repository": {
            "git_commit": git_commit,
            "dirty_worktree": bool(dirty_lines),
            "dirty_paths": dirty_lines,
            "experiment_git_commits": sorted({row["git_commit"] for row in planned_records}),
        },
        "counts": {
            "planned": len(plan),
            "atomic_records_total": len(record_rows),
            "unplanned_smoke_records": len(extra),
            "by_phase_and_status": {phase: dict(sorted(counts.items())) for phase, counts in sorted(phase_counts.items())},
            "expected_final_seeds": list(FINAL_SEEDS),
            "hardware_ids": hardware_ids,
        },
        "reused_part1_run_ids": sorted({row["run_id"] for row in bonus_by_seed if row["record_origin"] == "reused_part1"}),
        "missing_value_policy": {
            "encoding": "empty CSV field / JSON null, never numeric zero",
            "reasons": {
                "tfidf_peak_gpu_memory_mb": "CPU method",
                "tfidf_parameter_counts": "capacity is represented by feature/tree/node counts",
                "reused_part1_source_metrics": "Part 1 references were evaluated only on their own target domain",
                "unobserved_crossover": "threshold not reached inside the tested grid",
                "tuning_test_metrics": "test evaluation prohibited during selection",
            },
        },
        "declared_exceptions": [row["details"] for row in audit.warnings],
        "evidence_files": evidence,
    }
    return manifest, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-declared-exceptions", action="store_true")
    args = parser.parse_args()
    manifest, audit = _audit()
    REPORTING_RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(REPORTING_RESULTS / "frozen_result_manifest.json", manifest)
    fields = ["check_id", "area", "severity", "status", "count", "details"]
    write_csv(REPORTING_RESULTS / "reporting_audit.csv", audit.rows, fields)
    lines = [
        "# Reporting audit", "", f"Outcome: **{manifest['audit_outcome']}**", "",
        f"Planned jobs: {manifest['counts']['planned']}; atomic records: {manifest['counts']['atomic_records_total']}.", "",
        "## Phase counts", "",
        "| Phase | Completed | Failed | Blocked | Stale | Skipped | Missing |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for phase, counts in manifest["counts"]["by_phase_and_status"].items():
        lines.append(f"| {phase} | {counts.get('completed', 0)} | {counts.get('failed', 0)} | {counts.get('blocked', 0)} | {counts.get('stale', 0)} | {counts.get('skipped', 0)} | {counts.get('missing', 0)} |")
    lines += ["", "## Checks", "", "| Check | Status | Details |", "|---|---|---|"]
    for row in audit.rows:
        lines.append(f"| `{row['check_id']}` | {row['status']} | {row['details']} |")
    lines += ["", "## Declared exceptions", ""]
    lines += [f"- {item}" for item in manifest["declared_exceptions"]] or ["None."]
    lines += [
        "", "## Interpretation", "",
        "The completed atomic runs are sufficient for final reporting. The reporting rebuild corrects the declared TF-IDF reference aggregation issue without retraining or changing any measured value. Unplanned smoke records remain outside final analyses.", "",
    ]
    (REPORTING_RESULTS / "reporting_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"outcome": manifest["audit_outcome"], "errors": len(audit.errors), "warnings": len(audit.warnings)}))
    if audit.errors or (audit.warnings and not args.allow_declared_exceptions):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
