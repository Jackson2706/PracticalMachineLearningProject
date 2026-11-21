from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Source"
RESULTS = SOURCE / "results"
REPORTING_RESULTS = RESULTS / "reporting"
SUBMISSION = ROOT / "submission"
FINAL_SEEDS = (13, 42, 2026)
DATASETS = ("nsmc", "shopping", "steam")
TFIDF_MODELS = ("tfidf_lr", "tfidf_svm", "tfidf_rf", "tfidf_xgb")
NEURAL_MODELS = ("avg_mlp", "lstm", "gru", "transformer", "mamba")

DISPLAY = {
    "nsmc": "NSMC",
    "shopping": "Shopping",
    "steam": "Steam",
    "tfidf_lr": "TF-IDF + LR",
    "tfidf_svm": "TF-IDF + SVM",
    "tfidf_rf": "TF-IDF + RF",
    "tfidf_xgb": "TF-IDF + XGBoost",
    "avg_mlp": "Avg. embeddings + MLP",
    "lstm": "LSTM",
    "gru": "GRU",
    "transformer": "Transformer",
    "mamba": "Mamba",
    "best_tfidf": "Best TF-IDF",
    "target_transformer": "Target-only Transformer",
    "pooled_ce": "Pooled CE",
    "supcon": "SupCon",
    "db_cdsupcon": "DB-CDSupCon",
}

# Okabe-Ito plus neutral gray; stable across every generated figure.
COLORS = {
    "tfidf_lr": "#0072B2",
    "tfidf_svm": "#56B4E9",
    "tfidf_rf": "#009E73",
    "tfidf_xgb": "#F0E442",
    "avg_mlp": "#E69F00",
    "lstm": "#D55E00",
    "gru": "#CC79A7",
    "transformer": "#000000",
    "mamba": "#7F7F7F",
    "best_tfidf": "#0072B2",
    "target_transformer": "#7F7F7F",
    "pooled_ce": "#009E73",
    "supcon": "#E69F00",
    "db_cdsupcon": "#CC79A7",
}

MARKERS = {
    "tfidf_lr": "o", "tfidf_svm": "s", "tfidf_rf": "^", "tfidf_xgb": "D",
    "avg_mlp": "P", "lstm": "v", "gru": "X", "transformer": "*", "mamba": "h",
    "best_tfidf": "s", "target_transformer": "o", "pooled_ce": "^",
    "supcon": "D", "db_cdsupcon": "P",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def as_int(value: Any) -> int | None:
    return None if value in (None, "") else int(float(value))


def mean_sd(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else None


def normalized_log_auc(sizes: Sequence[int], scores: Sequence[float]) -> float:
    if len(sizes) != len(scores) or len(sizes) < 2 or any(size <= 0 for size in sizes):
        raise ValueError("AULC needs at least two positive, aligned training sizes")
    order = np.argsort(np.asarray(sizes))
    x = np.log2(np.asarray(sizes, dtype=float)[order])
    y = np.asarray(scores, dtype=float)[order]
    if len(np.unique(x)) != len(x):
        raise ValueError("training sizes must be unique")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def first_threshold(sizes: Sequence[int], differences: Sequence[float], margin: float = 0.01) -> int | None:
    return next((size for size, difference in sorted(zip(sizes, differences)) if difference >= margin), None)


def first_sustained_threshold(
    sizes: Sequence[int], differences: Sequence[float], margin: float = 0.01
) -> int | None:
    pairs = sorted(zip(sizes, differences))
    return next((size for index, (size, value) in enumerate(pairs) if value >= margin and all(v >= margin for _, v in pairs[index:])), None)


def labels_saved(reference: int | None, candidate: int | None) -> int | None:
    return None if reference is None or candidate is None else max(0, reference - candidate)


def seed_complete(seeds: Iterable[int], expected: Sequence[int] = FINAL_SEEDS) -> bool:
    return set(seeds) == set(expected)


def best_complete(
    rows: Sequence[dict[str, Any]], score_key: str, *, complete_key: str = "complete", name_key: str = "model"
) -> dict[str, Any] | None:
    candidates = [row for row in rows if bool(row.get(complete_key))]
    return max(candidates, key=lambda row: (float(row[score_key]), str(row[name_key])), default=None)


def pareto_mask(quality: Sequence[float], cost: Sequence[float]) -> list[bool]:
    points = list(zip(quality, cost))
    return [
        not any((q2 >= q and c2 <= c) and (q2 > q or c2 < c) for q2, c2 in points)
        for q, c in points
    ]


def latex_escape(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&",
        "#": r"\#", "{": r"\{", "}": r"\}", "$": r"\$", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def format_mean_sd(mean: float | None, sd: float | None, n: int, *, percent: bool = True) -> str:
    if mean is None:
        return r"\textit{missing}"
    scale = 100 if percent else 1
    decimals = 2 if percent else 3
    sd_text = "--" if sd is None else f"{sd * scale:.{decimals}f}"
    suffix = "" if n == len(FINAL_SEEDS) else r"\textsuperscript{!}"
    return f"{mean * scale:.{decimals}f} $\\pm$ {sd_text} ({n}){suffix}"

