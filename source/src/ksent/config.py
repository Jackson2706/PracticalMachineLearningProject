"""Configuration loading and canonical paths.

All paths are resolved relative to the ``source/`` directory so that scripts work
regardless of the current working directory.
"""
from __future__ import annotations

import functools
import os
import re
from pathlib import Path
from typing import Any

import yaml

SOURCE_DIR: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = SOURCE_DIR / "configs"

TFIDF_METHODS: tuple[str, ...] = ("tfidf_lr", "tfidf_svm", "tfidf_rf", "tfidf_xgb")
NEURAL_METHODS: tuple[str, ...] = ("emb_mlp", "emb_rnn_uni", "emb_lstm_uni", "emb_gru_uni",
                                  "emb_rnn", "emb_lstm", "emb_gru", "small_transformer", "small_mamba")
ALL_METHODS: tuple[str, ...] = TFIDF_METHODS + NEURAL_METHODS

METHOD_DISPLAY: dict[str, str] = {
    "tfidf_lr": "TF-IDF + LR",
    "tfidf_svm": "TF-IDF + SVM",
    "tfidf_rf": "TF-IDF + RF",
    "tfidf_xgb": "TF-IDF + XGBoost",
    "emb_mlp": "Emb + MLP",
    "emb_rnn_uni": "Emb + RNN",
    "emb_lstm_uni": "Emb + LSTM",
    "emb_gru_uni": "Emb + GRU",
    "emb_rnn": "Emb + BiRNN",
    "emb_lstm": "Emb + BiLSTM",
    "emb_gru": "Emb + BiGRU",
    "small_transformer": "Transformer",
    "small_mamba": "Mamba",
}
DATASET_DISPLAY: dict[str, str] = {"nsmc": "NSMC", "shopping": "Shopping", "steam": "Steam"}


def family(method: str) -> str:
    """Return ``"tfidf"`` or ``"neural"`` for a method name."""
    if method in TFIDF_METHODS:
        return "tfidf"
    if method in NEURAL_METHODS:
        return "neural"
    raise KeyError(f"unknown method {method!r}")


def load_yaml(path: str | Path) -> Any:
    """Load a YAML file (returns ``None`` for an empty file)."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(obj: Any, path: str | Path) -> None:
    """Write ``obj`` as YAML, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


@functools.lru_cache(maxsize=1)
def global_cfg() -> dict[str, Any]:
    """Return the parsed ``configs/global.yaml``."""
    return load_yaml(CONFIG_DIR / "global.yaml")


@functools.lru_cache(maxsize=1)
def data_cfg() -> dict[str, Any]:
    """Return the parsed ``configs/data.yaml``."""
    return load_yaml(CONFIG_DIR / "data.yaml")


# Optional overrides (used by tests / dry runs of the analysis on smoke results).
_PATH_ENV: dict[str, str] = {"results": "KSENT_RESULTS_DIR", "report": "KSENT_REPORT_DIR"}


def path(key: str) -> Path:
    """Absolute path for an entry of ``global.yaml: paths`` (``results`` and ``report``
    can be redirected with ``KSENT_RESULTS_DIR`` / ``KSENT_REPORT_DIR``)."""
    env = _PATH_ENV.get(key)
    if env and os.environ.get(env):
        return Path(os.environ[env]).resolve()
    return (SOURCE_DIR / global_cfg()["paths"][key]).resolve()


def processed_dir(dataset: str) -> Path:
    """Directory holding the processed splits of ``dataset``."""
    return path("data_processed") / dataset


def search_space(method: str) -> dict[str, Any]:
    """Return the 10-config search space of ``method``."""
    space = load_yaml(path("search_spaces") / f"{method}.yaml")
    if len(space["configs"]) != 10:
        raise ValueError(f"{method}: search space must contain exactly 10 configs")
    return space


def resolve_config(method: str, config_id: int) -> dict[str, Any]:
    """Merge the ``fixed`` block of a search space with config ``config_id``."""
    space = search_space(method)
    cfg = dict(space.get("fixed") or {})
    entry = next(c for c in space["configs"] if c["id"] == config_id)
    cfg.update({k: v for k, v in entry.items() if k != "id"})
    cfg["config_id"] = config_id
    return cfg


def frozen_path(dataset: str, method: str) -> Path:
    """Path of the frozen (tuned) config of ``(dataset, method)``."""
    return path("frozen") / dataset / f"{method}.yaml"


def frozen_config(dataset: str, method: str) -> dict[str, Any]:
    """Load the frozen config written by :mod:`ksent.tune`."""
    p = frozen_path(dataset, method)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run scripts/04_tune.sh first")
    return load_yaml(p)["params"]


FULL_SIZE = "full"
_NEAR_FULL = 0.9  # numeric sizes above this fraction of the pool are covered by "full"


@functools.lru_cache(maxsize=None)
def pool_size(dataset: str) -> int:
    """Number of reviews available for training (``train_pool`` + ``train_extra``)."""
    import pyarrow.parquet as pq

    d = processed_dir(dataset)
    return sum(pq.ParquetFile(d / f"{s}.parquet").metadata.num_rows
               for s in ("train_pool", "train_extra") if (d / f"{s}.parquet").exists())


def resolve_size(dataset: str, size: int | str) -> int:
    """Concrete training size; ``"full"`` is every available training review of ``dataset``."""
    return pool_size(dataset) if str(size) == FULL_SIZE else int(size)


def resolve_sizes(dataset: str, sizes: list | None = None) -> list[int]:
    """``global.yaml: sizes`` (or ``sizes``) for ``dataset``, ascending, ``full`` resolved.
    Sizes above the available pool are dropped, and so are sizes within 10% below it
    (e.g. 140k for NSMC's 143,640): they would nearly duplicate the ``full`` run."""
    n = pool_size(dataset)
    raw = global_cfg()["sizes"] if sizes is None else sizes
    if not n:  # data not prepared (e.g. analysis tests): numeric sizes only
        return sorted({int(x) for x in raw if str(x) != FULL_SIZE})
    return sorted({s for s in (resolve_size(dataset, x) for x in raw) if s == n or s <= _NEAR_FULL * n})


def size_label(dataset: str, size: int) -> str:
    """``"full"`` for the full pool of ``dataset``, else the size with thousands separators."""
    return FULL_SIZE if FULL_SIZE in map(str, global_cfg()["sizes"]) and int(size) == pool_size(dataset) \
        else f"{int(size):,}"


def main_tokenizer() -> str:
    """The main-experiment tokenizer (``kiwi``)."""
    return str(global_cfg()["tokenizer"])


def max_seq_len(tokenizer: str) -> int:
    """Max neural sequence length of ``tokenizer`` stored in ``global.yaml``."""
    value = (global_cfg().get("max_seq_len") or {}).get(tokenizer)
    if not value:
        raise RuntimeError(f"max_seq_len[{tokenizer}] is null - run scripts/01_prepare_data.sh first")
    return int(value)


def write_max_seq_len(values: dict[str, int]) -> None:
    """Persist the per-tokenizer ``max_seq_len`` mapping into ``global.yaml`` (keeps comments)."""
    p = CONFIG_DIR / "global.yaml"
    text = p.read_text(encoding="utf-8")
    inner = ", ".join(f"{k}: {int(v)}" for k, v in values.items())
    text = re.sub(r"^max_seq_len:.*$", "max_seq_len: {" + inner + "}", text, flags=re.M)
    p.write_text(text, encoding="utf-8")
    global_cfg.cache_clear()
