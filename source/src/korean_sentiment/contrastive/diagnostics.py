from __future__ import annotations

from typing import Any

import numpy as np


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / max(denominator, 1e-12))


def representation_diagnostics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    pair_count: int,
    seed: int,
) -> dict[str, Any]:
    embeddings = np.asarray(embeddings, dtype=float)
    labels = np.asarray(labels)
    domains = np.asarray(domains)
    if not (len(embeddings) == len(labels) == len(domains)):
        raise ValueError("embedding, label, and domain lengths differ")
    generator = np.random.default_rng(seed)
    categories: dict[str, list[float]] = {
        "same_sentiment_cross_domain": [],
        "different_sentiment_cross_domain": [],
        "same_domain_same_sentiment": [],
    }
    possible: dict[str, list[tuple[int, int]]] = {name: [] for name in categories}
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            if domains[left] != domains[right] and labels[left] == labels[right]:
                possible["same_sentiment_cross_domain"].append((left, right))
            elif domains[left] != domains[right] and labels[left] != labels[right]:
                possible["different_sentiment_cross_domain"].append((left, right))
            elif domains[left] == domains[right] and labels[left] == labels[right]:
                possible["same_domain_same_sentiment"].append((left, right))
    for name, pairs in possible.items():
        if not pairs:
            continue
        selected = generator.choice(len(pairs), size=min(pair_count, len(pairs)), replace=False)
        categories[name] = [_cosine(embeddings[pairs[index][0]], embeddings[pairs[index][1]]) for index in selected]
    means = {name: (float(np.mean(values)) if values else None) for name, values in categories.items()}
    same = means["same_sentiment_cross_domain"]
    different = means["different_sentiment_cross_domain"]
    return {
        "same_sentiment_cross_domain_cosine": same,
        "different_sentiment_cross_domain_cosine": different,
        "cross_domain_sentiment_separation_gap": None if same is None or different is None else same - different,
        "same_domain_same_sentiment_cosine": means["same_domain_same_sentiment"],
        "representation_pair_count": sum(len(values) for values in categories.values()),
        "representation_pair_counts": {name: len(values) for name, values in categories.items()},
        "representation_pair_seed": seed,
    }


def normalized_log_auc(sizes: list[int], scores: list[float]) -> float:
    if len(sizes) != len(scores) or len(sizes) < 2:
        raise ValueError("at least two aligned points are required")
    order = np.argsort(sizes)
    x = np.log2(np.asarray(sizes, dtype=float)[order])
    y = np.asarray(scores, dtype=float)[order]
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(y, x) / (x[-1] - x[0]))


def label_savings(
    reference_sizes: list[int],
    reference_scores: list[float],
    candidate_sizes: list[int],
    candidate_scores: list[float],
    tfidf_scores: dict[int, float],
    margin: float = 0.01,
) -> dict[str, Any]:
    def threshold(sizes: list[int], scores: list[float]) -> int | None:
        for size, score in sorted(zip(sizes, scores)):
            baseline = tfidf_scores.get(size)
            if baseline is not None and score >= baseline + margin:
                return size
        return None

    reference = threshold(reference_sizes, reference_scores)
    candidate = threshold(candidate_sizes, candidate_scores)
    return {
        "reference_threshold": reference,
        "candidate_threshold": candidate,
        "target_training_labels_saved": None if reference is None or candidate is None else max(0, reference - candidate),
        "status": "observed" if reference is not None and candidate is not None else "not_observed_within_tested_sizes",
    }
