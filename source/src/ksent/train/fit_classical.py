"""Fit a TF-IDF pipeline on the current training subset only (always on CPU)."""
from __future__ import annotations

from typing import Any

import numpy as np

from ksent.config import global_cfg
from ksent.eval.cost import Timer
from ksent.features.tfidf import make_vectorizer
from ksent.models.classical import build_classifier
from ksent.pipeline import TfidfPipeline


def fit_classical(method: str, params: dict[str, Any], train_inputs: list, y_train: np.ndarray,
                  seed: int, n_threads: int, tokenizer: str, tuning: bool = False,
                  val_inputs: list | None = None, y_val: np.ndarray | None = None) -> tuple[TfidfPipeline, dict]:
    """Fit vectoriser + classifier; return the pipeline and timing/extra info.

    ``train_inputs`` are token lists (kiwi/whitespace) or normalized strings (char).
    In tuning mode XGBoost grows up to ``xgb_tuning.max_rounds`` trees with early
    stopping on the validation set; the best round count is returned in
    ``info["xgb_best_rounds"]`` and frozen by :mod:`ksent.tune`.
    """
    info: dict[str, Any] = {}
    with Timer() as t_vec:
        vec = make_vectorizer(tokenizer)
        X = vec.fit_transform(train_inputs)
    info["featurize_time_s"] = t_vec.seconds
    info["n_features"] = int(X.shape[1])
    fit_kw: dict[str, Any] = {}
    if method == "tfidf_xgb" and tuning:
        xt = global_cfg()["xgb_tuning"]
        clf = build_classifier(method, params, seed, n_threads, n_estimators=int(xt["max_rounds"]))
        clf.set_params(early_stopping_rounds=int(xt["early_stopping_rounds"]))
        fit_kw = {"eval_set": [(vec.transform(val_inputs), y_val)], "verbose": False}
    else:
        clf = build_classifier(method, params, seed, n_threads)
    with Timer() as t_fit:
        clf.fit(X, y_train, **fit_kw)
    info["classifier_time_s"] = t_fit.seconds
    if method == "tfidf_xgb" and tuning:
        info["xgb_best_rounds"] = int(clf.best_iteration) + 1
        info["xgb_rounds_grown"] = int(clf.get_booster().num_boosted_rounds())
    return TfidfPipeline(method, vec, clf, tokenizer), info
