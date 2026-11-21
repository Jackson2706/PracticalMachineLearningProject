# Reporting audit

Outcome: **complete_with_declared_exceptions**

Planned jobs: 1251; atomic records: 1263.

## Phase counts

| Phase | Completed | Failed | Blocked | Stale | Skipped | Missing |
|---|---:|---:|---:|---:|---:|---:|
| bonus_final | 108 | 0 | 0 | 0 | 0 | 0 |
| bonus_tuning | 36 | 0 | 0 | 0 | 0 | 0 |
| part1_final | 837 | 0 | 0 | 0 | 0 | 0 |
| part1_tuning | 270 | 0 | 0 | 0 | 0 | 0 |

## Checks

| Check | Status | Details |
|---|---|---|
| `plan_count` | pass | planned=1251, expected=1251 |
| `unique_plan_run_ids` | pass | plan run IDs are unique |
| `unique_record_run_ids` | pass | duplicate record IDs=0 |
| `planned_records_present` | pass | missing planned records=0 |
| `unplanned_records_declared` | pass | unplanned records=12 (smoke records; excluded from final analysis) |
| `all_planned_jobs_completed` | pass | non-completed groups=[] |
| `final_seed_completeness` | pass | incomplete conditions=0 |
| `no_test_metrics_in_tuning` | pass | tuning rows with test metrics=0 |
| `part1_selection_recomputed` | pass | selector mismatches=0 |
| `part1_configs_frozen` | pass | final config mismatches=0 |
| `bonus_selection_recomputed` | pass | selector mismatches=0 |
| `shared_data_manifest` | pass | run data-hash mismatches=0 |
| `nested_subsets_and_no_leakage` | pass | subset/leakage issues=0 |
| `identical_part1_subset_ids` | pass | all Part 1 models reference the same dataset/size/seed prefix manifests |
| `bonus_matched_design` | pass | mismatched bonus cells=0 |
| `bonus_tfidf_reference_selection` | warning | existing bonus aggregate mixes per-seed TF-IDF winners in 7 rows; reporting rebuild selects one model by three-seed mean |
| `metric_ranges` | pass | invalid final metric rows=0 |
| `confusion_totals` | pass | confusion total mismatches=0 |
| `resource_plausibility` | pass | invalid resource rows=0 |
| `atomic_artifacts` | pass | runs missing required artifacts=0 |
| `timing_hardware_identity` | pass | hardware IDs=['8b5663550ab8bee2'] |

## Declared exceptions

- existing bonus aggregate mixes per-seed TF-IDF winners in 7 rows; reporting rebuild selects one model by three-seed mean

## Interpretation

The completed atomic runs are sufficient for final reporting. The reporting rebuild corrects the declared TF-IDF reference aggregation issue without retraining or changing any measured value. Unplanned smoke records remain outside final analyses.
