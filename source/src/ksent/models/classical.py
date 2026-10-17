"""TF-IDF classifiers (scikit-learn / XGBoost) and their model-size statistics."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def build_classifier(method: str, params: dict[str, Any], seed: int, n_threads: int, n_estimators: int | None = None):
    """Instantiate the classifier of ``method`` with hyperparameters ``params``.

    ``n_estimators`` overrides the XGBoost round count (used during tuning, where
    up to ``max_rounds`` are grown with early stopping on validation).
    """
    p = {k: v for k, v in params.items() if k not in ("config_id",)}
    if method == "tfidf_lr":
        return LogisticRegression(C=float(p["C"]), solver=p.get("solver", "liblinear"),
                                  max_iter=int(p.get("max_iter", 2000)), random_state=seed)
    if method == "tfidf_svm":
        return LinearSVC(C=float(p["C"]), max_iter=int(p.get("max_iter", 10000)),
                         dual=p.get("dual", "auto"), random_state=seed)
    if method == "tfidf_rf":
        return RandomForestClassifier(
            n_estimators=int(p["n_estimators"]), max_depth=p.get("max_depth"),
            max_features=p.get("max_features", "sqrt"), min_samples_leaf=int(p.get("min_samples_leaf", 1)),
            class_weight=p.get("class_weight"), n_jobs=n_threads, random_state=seed,
        )
    if method == "tfidf_xgb":
        from xgboost import XGBClassifier

        rounds = int(n_estimators if n_estimators is not None else p["n_estimators"])
        return XGBClassifier(
            n_estimators=rounds, max_depth=int(p["max_depth"]), learning_rate=float(p["learning_rate"]),
            subsample=float(p["subsample"]), colsample_bytree=float(p["colsample_bytree"]),
            tree_method=p.get("tree_method", "hist"), device="cpu", eval_metric=p.get("eval_metric", "logloss"),
            n_jobs=n_threads, random_state=seed, verbosity=0,
        )
    raise KeyError(method)


def model_size_stats(method: str, clf) -> dict[str, Any]:
    """Size statistics: nonzero coefficients (linear), node counts (trees)."""
    if method in ("tfidf_lr", "tfidf_svm"):
        coef = np.asarray(clf.coef_)
        return {"n_coef": int(coef.size), "n_nonzero_coef": int(np.count_nonzero(coef))}
    if method == "tfidf_rf":
        nodes = [est.tree_.node_count for est in clf.estimators_]
        return {"n_trees": len(nodes), "total_nodes": int(sum(nodes)),
                "mean_depth": float(np.mean([est.tree_.max_depth for est in clf.estimators_]))}
    if method == "tfidf_xgb":
        dump = clf.get_booster().get_dump()
        return {"n_trees": len(dump), "total_nodes": int(sum(t.count("\n") for t in dump))}
    return {}


def predict_scores(method: str, clf, X) -> tuple[np.ndarray, np.ndarray | None]:
    """Predicted labels and P(positive) when the classifier provides probabilities."""
    pred = np.asarray(clf.predict(X)).astype(int)
    if hasattr(clf, "predict_proba"):
        return pred, np.asarray(clf.predict_proba(X))[:, 1]
    return pred, None
