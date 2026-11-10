from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from korean_sentiment.evaluation.metrics import classification_metrics
from korean_sentiment.models import ModelBlocked, load_pipeline, make_pipeline
from korean_sentiment.models.neural import AverageMLP, RecurrentClassifier, SmallMamba, SmallTransformer


BASE = {"embedding_dim": 16, "hidden_dim": 16, "dropout": 0.1, "num_layers": 1}


@pytest.mark.parametrize(
    "model",
    [
        AverageMLP(30, BASE),
        RecurrentClassifier(30, BASE, "lstm"),
        RecurrentClassifier(30, BASE, "gru"),
        SmallTransformer(30, {**BASE, "num_heads": 4, "ff_multiplier": 2}, 10),
    ],
)
def test_neural_forward_shapes(model):
    ids = torch.randint(0, 30, (4, 10))
    lengths = torch.tensor([10, 8, 5, 2])
    logits = model(ids, lengths)
    assert logits.shape == (4, 2)
    assert torch.isfinite(logits).all()


def test_mamba_is_explicitly_blocked_without_linux_cuda():
    config = {**BASE, "d_state": 8, "d_conv": 4, "expand": 2}
    with pytest.raises(RuntimeError, match="mamba requires Linux"):
        SmallMamba(30, config, 10)


@pytest.mark.parametrize(
    ("name", "config"),
    [
        ("tfidf_lr", {"C": 1.0, "solver": "liblinear", "max_iter": 200}),
        ("tfidf_svm", {"C": 1.0}),
        ("tfidf_rf", {"n_estimators": 5, "max_depth": 3, "max_features": "sqrt", "min_samples_leaf": 1}),
        ("tfidf_xgb", {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.2, "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1.0}),
    ],
)
def test_classical_tiny_fit_predict_save_load(name, config, binary_texts, tmp_path: Path):
    train, labels, validation, validation_labels = binary_texts
    pipeline = make_pipeline(name, config, 13, "cpu", 1)
    pipeline.fit(train, labels, validation, validation_labels)
    prediction = pipeline.predict(validation)
    artifact = pipeline.save(tmp_path / name)
    loaded = load_pipeline(artifact)
    assert np.array_equal(prediction, loaded.predict(validation))
    assert set(prediction) <= {0, 1}


@pytest.mark.parametrize(
    ("name", "config"),
    [
        ("avg_mlp", BASE),
        ("lstm", BASE),
        ("gru", BASE),
        ("transformer", {**BASE, "num_heads": 4, "ff_multiplier": 2}),
    ],
)
def test_neural_tiny_fit_predict_save_load(name, config, binary_texts, tmp_path: Path):
    train, labels, validation, validation_labels = binary_texts
    pipeline = make_pipeline(name, {**config, "learning_rate": 0.001}, 13, "cpu", 1)
    result = pipeline.fit(train, labels, validation, validation_labels, max_epochs=1, patience=1)
    prediction = pipeline.predict(validation)
    artifact = pipeline.save(tmp_path / name)
    loaded = load_pipeline(artifact)
    assert np.array_equal(prediction, loaded.predict(validation))
    assert result["best_epoch"] == 1
    assert pipeline.metadata()["total_parameters"] > 0


def test_metrics_and_confusion_matrix():
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (1, 1, 0, 2)
    assert metrics["accuracy"] == pytest.approx(0.75)
