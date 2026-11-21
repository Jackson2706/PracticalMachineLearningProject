from __future__ import annotations

import math

from reporting_utils import (
    best_complete,
    first_sustained_threshold,
    first_threshold,
    labels_saved,
    normalized_log_auc,
    seed_complete,
)


def test_crossover_observed_and_sustained() -> None:
    sizes = [200, 500, 1000, 2000]
    gaps = [0.0, 0.012, 0.008, 0.015]
    assert first_threshold(sizes, gaps) == 500
    assert first_sustained_threshold(sizes, gaps) == 2000
    assert first_sustained_threshold(sizes, [0.0, 0.009, 0.008, 0.007]) is None


def test_label_savings_requires_both_thresholds() -> None:
    assert labels_saved(2000, 500) == 1500
    assert labels_saved(500, 1000) == 0
    assert labels_saved(None, 500) is None
    assert labels_saved(2000, None) is None


def test_normalized_log_auc() -> None:
    assert math.isclose(normalized_log_auc([1, 2, 4], [0.2, 0.4, 0.6]), 0.4)


def test_seed_completeness() -> None:
    assert seed_complete([2026, 13, 42])
    assert not seed_complete([13, 42])
    assert not seed_complete([13, 42, 2026, 777])


def test_best_method_uses_only_complete_rows() -> None:
    rows = [
        {"model": "complete", "macro_f1": 0.8, "complete": True},
        {"model": "incomplete", "macro_f1": 0.9, "complete": False},
    ]
    assert best_complete(rows, "macro_f1")["model"] == "complete"
    assert best_complete([], "macro_f1") is None
