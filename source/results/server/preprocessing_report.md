# Preprocessing

```json
{
  "statistics": [
    {
      "dataset": "nsmc",
      "train_rows": "130797",
      "validation_rows": "14534",
      "test_rows": "49110",
      "n_full": "130797",
      "training_sizes": "[200, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 130797]"
    },
    {
      "dataset": "shopping",
      "train_rows": "159918",
      "validation_rows": "19990",
      "test_rows": "19990",
      "n_full": "159918",
      "training_sizes": "[200, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 159918]"
    },
    {
      "dataset": "steam",
      "train_rows": "79828",
      "validation_rows": "9979",
      "test_rows": "9979",
      "n_full": "79828",
      "training_sizes": "[200, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 79828]"
    }
  ],
  "cleaning": {
    "steam_cleaning_report": {
      "dataset": "steam",
      "n_full": 79828,
      "schedule": [
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
      ],
      "sources": {
        "all": {
          "clean_rows": 99786,
          "conflicting_keys_removed": 107,
          "conflicting_occurrences_removed": 214,
          "duplicate_pairs_removed": 0,
          "empty_removed": 0,
          "malformed_rows": 0,
          "negative_rows": 49897,
          "neutral_rating_rows": 0,
          "parsed_rows": 100000,
          "positive_rows": 49889,
          "raw_rows": 100000
        }
      },
      "splits": {
        "test_rows": 9979,
        "train_rows": 79828,
        "validation_rows": 9979
      }
    },
    "nsmc_cleaning_report": {
      "dataset": "nsmc",
      "n_full": 130797,
      "schedule": [
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
      "sources": {
        "test": {
          "clean_rows": 49110,
          "conflicting_keys_removed": 47,
          "conflicting_occurrences_removed": 306,
          "duplicate_pairs_removed": 581,
          "empty_removed": 3,
          "malformed_rows": 0,
          "negative_rows": 24423,
          "neutral_rating_rows": 0,
          "parsed_rows": 50000,
          "positive_rows": 24687,
          "raw_rows": 50000
        },
        "train": {
          "clean_rows": 146025,
          "conflicting_keys_removed": 157,
          "conflicting_occurrences_removed": 1451,
          "duplicate_pairs_removed": 2519,
          "empty_removed": 5,
          "malformed_rows": 0,
          "negative_rows": 73272,
          "neutral_rating_rows": 0,
          "parsed_rows": 150000,
          "positive_rows": 72753,
          "raw_rows": 150000
        }
      },
      "splits": {
        "official_cross_partition_train_rows_removed": 694,
        "test_rows": 49110,
        "train_rows": 130797,
        "validation_rows": 14534
      }
    },
    "shopping_cleaning_report": {
      "dataset": "shopping",
      "n_full": 159918,
      "schedule": [
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
      "sources": {
        "all": {
          "clean_rows": 199898,
          "conflicting_keys_removed": 10,
          "conflicting_occurrences_removed": 20,
          "duplicate_pairs_removed": 82,
          "empty_removed": 0,
          "malformed_rows": 0,
          "negative_rows": 99950,
          "neutral_rating_rows": 0,
          "parsed_rows": 200000,
          "positive_rows": 99948,
          "raw_rows": 200000
        }
      },
      "splits": {
        "test_rows": 19990,
        "train_rows": 159918,
        "validation_rows": 19990
      }
    }
  },
  "split_class_balance": {
    "nsmc": {
      "train": {
        "0": 65615,
        "1": 65182
      },
      "validation": {
        "0": 7291,
        "1": 7243
      },
      "test": {
        "0": 24423,
        "1": 24687
      }
    },
    "shopping": {
      "train": {
        "0": 79960,
        "1": 79958
      },
      "validation": {
        "0": 9995,
        "1": 9995
      },
      "test": {
        "0": 9995,
        "1": 9995
      }
    },
    "steam": {
      "train": {
        "0": 39917,
        "1": 39911
      },
      "validation": {
        "0": 4990,
        "1": 4989
      },
      "test": {
        "0": 4990,
        "1": 4989
      }
    }
  }
}
```

Checksums: source/data/manifests/download_manifest.csv. Splits: source/data/splits/. Nested orders and data hash: source/data/manifests/. Bonus uses target prefixes 200/500/1000/2000 and source prefixes 2000 from each other domain.
