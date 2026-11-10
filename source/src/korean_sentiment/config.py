from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml

SOURCE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SOURCE_ROOT.parent
CONFIG_DIR = SOURCE_ROOT / "configs"

DATASETS = ("nsmc", "shopping", "steam")
TFIDF_MODELS = ("tfidf_lr", "tfidf_svm", "tfidf_rf", "tfidf_xgb")
NEURAL_MODELS = ("avg_mlp", "lstm", "gru", "transformer", "mamba")
MODELS = TFIDF_MODELS + NEURAL_MODELS
FINAL_SEEDS = (13, 42, 2026)
TUNING_SEED = 777


def _path_override(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    path = Path(value).expanduser()
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def data_dir() -> Path:
    return _path_override("KS_DATA_DIR", SOURCE_ROOT / "data")


def results_dir() -> Path:
    return _path_override("KS_RESULTS_DIR", SOURCE_ROOT / "results")


def logs_dir() -> Path:
    return SOURCE_ROOT / "logs"


def checkpoints_dir() -> Path:
    return SOURCE_ROOT / "checkpoints"


@functools.lru_cache(maxsize=None)
def load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def dataset_config() -> dict[str, Any]:
    return load_yaml("datasets.yaml")


def experiment_config() -> dict[str, Any]:
    return load_yaml("experiment.yaml")


def bonus_config() -> dict[str, Any]:
    return load_yaml("bonus.yaml")


def model_configs(model: str) -> list[dict[str, Any]]:
    if model not in MODELS:
        raise KeyError(f"unknown model: {model}")
    values = load_yaml(f"models/{model}.yaml").get("configs", [])
    if len(values) != 10:
        raise ValueError(f"{model} must define exactly ten configurations, found {len(values)}")
    ids = [str(value.get("id")) for value in values]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{model} has duplicate configuration IDs")
    return [dict(value) for value in values]


def env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value
