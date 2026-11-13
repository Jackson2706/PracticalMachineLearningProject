# Plan verification

{
  "counts": {
    "part1_tuning": 270,
    "part1_final": 837,
    "bonus_tuning": 36,
    "bonus_final": 108
  },
  "total": 1251,
  "part1": {
    "dataset_full_sizes": {
      "nsmc": 130797,
      "shopping": 159918,
      "steam": 79828
    },
    "dataset_schedules": {
      "nsmc": [
        200,
        500,
        1000,
        2000,
        4000,
        8000,
        16000,
        32000,
        64000,
        130797
      ],
      "shopping": [
        200,
        500,
        1000,
        2000,
        4000,
        8000,
        16000,
        32000,
        64000,
        128000,
        159918
      ],
      "steam": [
        200,
        500,
        1000,
        2000,
        4000,
        8000,
        16000,
        32000,
        64000,
        79828
      ]
    },
    "difference_explanation": "matches the expected cleaned-pool schedules",
    "expected_final_jobs": 837,
    "expected_tuning_jobs": 270,
    "final_jobs": 837,
    "total_jobs": 1107,
    "tuning_jobs": 270
  }
}

Run IDs unique; final seeds 13,42,2026; 72 reference rows link existing Part 1 conditions without new jobs. Training loaders use train-only manifests; tuning loaders omit tests.

Bonus dependencies have been checked before execution.
