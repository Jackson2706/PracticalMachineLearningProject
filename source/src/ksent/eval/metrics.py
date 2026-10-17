"""Classification metrics (all returned as fractions; tables show them x100)."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """Macro-F1 (primary), accuracy, macro-P/R, per-class F1 and the 2x2 confusion matrix.

    Confusion matrix rows are true labels ``[neg, pos]``, columns predicted labels.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    kw = dict(labels=[0, 1], zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", **kw)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", **kw)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", **kw)),
        "f1_per_class": [float(v) for v in f1_score(y_true, y_pred, average=None, **kw)],
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int).tolist(),
        "n": int(len(y_true)),
    }


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Macro-F1 over labels {0, 1}."""
    return float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0))
