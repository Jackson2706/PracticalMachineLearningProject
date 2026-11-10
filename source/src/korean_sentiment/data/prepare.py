from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from korean_sentiment.config import FINAL_SEEDS, TUNING_SEED, data_dir, dataset_config, results_dir
from korean_sentiment.utils.io import atomic_write_csv, atomic_write_json, canonical_hash, sha256_file

SPLIT_COLUMNS = ["example_id", "dataset", "split", "label", "text_hash", "source_row_id"]


def normalize_text(text: Any) -> str:
    text = unicodedata.normalize("NFC", str(text).replace("\r\n", "\n").replace("\r", "\n"))
    return " ".join(text.split())


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def parse_file(dataset: str, source: str, path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts = {"raw_rows": 0, "malformed_rows": 0, "neutral_rating_rows": 0}
    with path.open(encoding="utf-8-sig", errors="strict", newline=None) as handle:
        for line_number, line in enumerate(handle):
            parts = line.rstrip("\r\n").split("\t")
            if dataset == "nsmc" and line_number == 0:
                if [part.strip().lower() for part in parts] != ["id", "document", "label"]:
                    raise ValueError(f"{path}: unexpected NSMC header")
                continue
            counts["raw_rows"] += 1
            try:
                if dataset == "nsmc":
                    if len(parts) < 3:
                        raise ValueError
                    original_id, raw_text, raw_label = parts[0], "\t".join(parts[1:-1]), parts[-1]
                    label = int(raw_label.strip())
                    if label not in (0, 1):
                        raise ValueError
                elif dataset == "shopping":
                    if len(parts) < 2:
                        raise ValueError
                    rating, raw_text = int(parts[0].strip()), "\t".join(parts[1:])
                    if rating == 3:
                        counts["neutral_rating_rows"] += 1
                        continue
                    if rating not in (1, 2, 4, 5):
                        raise ValueError
                    label = int(rating >= 4)
                    original_id = str(line_number)
                elif dataset == "steam":
                    if len(parts) < 2:
                        raise ValueError
                    label, raw_text = int(parts[0].strip()), "\t".join(parts[1:])
                    if label not in (0, 1):
                        raise ValueError
                    original_id = str(line_number)
                else:
                    raise KeyError(dataset)
            except (ValueError, TypeError):
                counts["malformed_rows"] += 1
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "source": source,
                    "source_row_id": f"{source}:{line_number}",
                    "original_id": original_id,
                    "text_raw": raw_text,
                    "label": label,
                }
            )
    return pd.DataFrame(rows), counts


def clean_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = frame.copy()
    report = {"parsed_rows": len(frame)}
    frame["text"] = frame["text_raw"].map(normalize_text)
    nonempty = frame["text"].str.len().gt(0)
    report["empty_removed"] = int((~nonempty).sum())
    frame = frame.loc[nonempty].copy()
    frame["normalized_text_key"] = frame["text"].map(normalize_text)
    conflicting_keys = set(
        frame.groupby("normalized_text_key", sort=False)["label"].nunique().loc[lambda values: values > 1].index
    )
    conflict = frame["normalized_text_key"].isin(conflicting_keys)
    report["conflicting_occurrences_removed"] = int(conflict.sum())
    report["conflicting_keys_removed"] = len(conflicting_keys)
    frame = frame.loc[~conflict].copy()
    duplicate = frame.duplicated(["normalized_text_key", "label"], keep="first")
    report["duplicate_pairs_removed"] = int(duplicate.sum())
    frame = frame.loc[~duplicate].copy()
    frame["text_hash"] = frame["normalized_text_key"].map(text_hash)
    frame["example_id"] = frame.apply(
        lambda row: f"{row.dataset}:{row.source_row_id}:{row.text_hash[:12]}", axis=1
    )
    frame = frame.reset_index(drop=True)
    report["clean_rows"] = len(frame)
    report["negative_rows"] = int((frame["label"] == 0).sum())
    report["positive_rows"] = int((frame["label"] == 1).sum())
    return frame, report


def _split_stratified(frame: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    first, second = train_test_split(
        frame, test_size=test_size, random_state=seed, stratify=frame["label"], shuffle=True
    )
    return first.copy(), second.copy()


def make_fixed_splits(dataset: str, frames: dict[str, pd.DataFrame], seed: int) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    leakage_report: dict[str, Any] = {}
    if dataset == "nsmc":
        train, test = frames["train"].copy(), frames["test"].copy()
        overlap = train["text_hash"].isin(set(test["text_hash"]))
        leakage_report["official_cross_partition_train_rows_removed"] = int(overlap.sum())
        train = train.loc[~overlap].copy()
        train, validation = _split_stratified(train, 0.10, seed)
    else:
        train, remainder = _split_stratified(frames["all"], 0.20, seed)
        validation, test = _split_stratified(remainder, 0.50, seed)
    splits = {"train": train, "validation": validation, "test": test}
    for name, frame in splits.items():
        frame["split"] = name
        splits[name] = frame.sort_values("example_id").reset_index(drop=True)
    validate_splits(splits)
    leakage_report["train_rows"] = len(splits["train"])
    leakage_report["validation_rows"] = len(splits["validation"])
    leakage_report["test_rows"] = len(splits["test"])
    return splits, leakage_report


def validate_splits(splits: dict[str, pd.DataFrame]) -> None:
    names = list(splits)
    for name, frame in splits.items():
        if set(frame["label"].unique()) - {0, 1}:
            raise AssertionError(f"invalid labels in {name}")
        if frame["example_id"].duplicated().any() or frame["text_hash"].duplicated().any():
            raise AssertionError(f"duplicates inside {name}")
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if set(splits[left]["example_id"]) & set(splits[right]["example_id"]):
                raise AssertionError(f"example overlap: {left}/{right}")
            if set(splits[left]["text_hash"]) & set(splits[right]["text_hash"]):
                raise AssertionError(f"text leakage: {left}/{right}")


def size_schedule(n_full: int, candidates: list[int] | None = None) -> list[int]:
    if n_full <= 0:
        raise ValueError("n_full must be positive")
    candidates = candidates or list(dataset_config()["candidate_sizes"])
    sizes = sorted({int(size) for size in candidates if int(size) < n_full})
    if sizes and sizes[-1] >= 0.9 * n_full:
        sizes.pop()
    return sizes + [n_full]


def stratified_order(frame: pd.DataFrame, seed: int) -> list[str]:
    ordered = frame.sort_values("example_id").reset_index(drop=True)
    labels = ordered["label"].to_numpy()
    ids = ordered["example_id"].to_numpy()
    generator = np.random.default_rng(seed)
    keys = np.empty(len(ordered), dtype=float)
    for label in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == label)
        permutation = generator.permutation(len(indices))
        keys[indices[permutation]] = (np.arange(len(indices)) + generator.random(len(indices))) / len(indices)
    return ids[np.argsort(keys, kind="stable")].tolist()


def validate_nested_manifest(frame: pd.DataFrame, order: list[str], sizes: list[int]) -> None:
    if len(order) != len(frame) or set(order) != set(frame["example_id"]):
        raise AssertionError("subset order does not cover the full training pool")
    if len(order) != len(set(order)):
        raise AssertionError("subset order contains duplicates")
    previous: set[str] = set()
    for size in sizes:
        current = set(order[:size])
        if len(current) != size or not previous < current if previous else len(current) != size:
            raise AssertionError(f"invalid nested prefix at size {size}")
        previous = current


def _write_split(dataset: str, name: str, frame: pd.DataFrame) -> None:
    columns = ["example_id", "dataset", "split", "label", "text", "text_hash", "source_row_id"]
    path = data_dir() / "processed" / dataset / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[columns].to_csv(path, index=False)


def preprocess_dataset(dataset: str) -> dict[str, Any]:
    config = dataset_config()
    dataset_spec = config["datasets"][dataset]
    reports: dict[str, Any] = {"dataset": dataset, "sources": {}}
    frames: dict[str, pd.DataFrame] = {}
    for source, spec in dataset_spec["files"].items():
        path = data_dir() / "raw" / dataset / spec["name"]
        if not path.exists():
            raise FileNotFoundError(f"missing raw file {path}; run download or use the README manual fallback")
        parsed, parse_counts = parse_file(dataset, source, path)
        cleaned, clean_counts = clean_frame(parsed)
        frames[source] = cleaned
        reports["sources"][source] = {**parse_counts, **clean_counts}
    splits, leakage_report = make_fixed_splits(dataset, frames, int(config["split_seed"]))
    reports["splits"] = leakage_report
    reports["n_full"] = len(splits["train"])
    reports["schedule"] = size_schedule(len(splits["train"]), list(config["candidate_sizes"]))
    for name, frame in splits.items():
        _write_split(dataset, name, frame)
    manifest = pd.concat([frame[SPLIT_COLUMNS] for frame in splits.values()], ignore_index=True)
    manifest = manifest.sort_values(["split", "example_id"])
    manifest_path = data_dir() / "splits" / f"{dataset}_splits.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(manifest_path, manifest.to_dict("records"), SPLIT_COLUMNS)
    manifest_path.chmod(0o444)
    atomic_write_json(data_dir() / "manifests" / f"{dataset}_cleaning_report.json", reports)
    subset_files = []
    label_by_id = splits["train"].set_index("example_id")["label"].to_dict()
    for seed in (*FINAL_SEEDS, TUNING_SEED):
        order = stratified_order(splits["train"], seed)
        regeneration = stratified_order(splits["train"], seed)
        if order != regeneration:
            raise AssertionError("subset regeneration is unstable")
        validate_nested_manifest(splits["train"], order, reports["schedule"])
        rows = [
            {
                "dataset": dataset,
                "seed": seed,
                "rank": rank,
                "example_id": example_id,
                "label": int(label_by_id[example_id]),
            }
            for rank, example_id in enumerate(order, start=1)
        ]
        subset_path = data_dir() / "manifests" / f"subset_{dataset}_seed_{seed}.csv"
        atomic_write_csv(
            subset_path,
            rows,
            ["dataset", "seed", "rank", "example_id", "label"],
        )
        subset_path.chmod(0o444)
        subset_files.append(subset_path)
    return {"report": reports, "split_manifest": manifest, "subset_files": subset_files}


def preprocess_all() -> list[dict[str, Any]]:
    outputs = [preprocess_dataset(dataset) for dataset in dataset_config()["datasets"]]
    split_rows = pd.concat([output["split_manifest"] for output in outputs], ignore_index=True)
    combined_split_path = data_dir() / "manifests" / "split_manifest.csv"
    atomic_write_csv(combined_split_path, split_rows.to_dict("records"), SPLIT_COLUMNS)
    combined_split_path.chmod(0o444)
    def subset_rows():
        for output in outputs:
            for path in output["subset_files"]:
                with path.open(encoding="utf-8", newline="") as handle:
                    yield from csv.DictReader(handle)

    atomic_write_csv(
        data_dir() / "manifests" / "subset_manifest.csv",
        subset_rows(),
        ["dataset", "seed", "rank", "example_id", "label"],
    )
    (data_dir() / "manifests" / "subset_manifest.csv").chmod(0o444)
    statistics = []
    for output in outputs:
        report = output["report"]
        statistics.append(
            {
                "dataset": report["dataset"],
                "train_rows": report["splits"]["train_rows"],
                "validation_rows": report["splits"]["validation_rows"],
                "test_rows": report["splits"]["test_rows"],
                "n_full": report["n_full"],
                "training_sizes": json.dumps(report["schedule"]),
            }
        )
    atomic_write_csv(
        results_dir() / "csv" / "dataset_statistics.csv",
        statistics,
        ["dataset", "train_rows", "validation_rows", "test_rows", "n_full", "training_sizes"],
    )
    tracked = sorted((data_dir() / "splits").glob("*.csv")) + sorted((data_dir() / "manifests").glob("subset_*.csv"))
    manifest_hash = canonical_hash([(str(path.relative_to(data_dir())), sha256_file(path)) for path in tracked])
    atomic_write_json(data_dir() / "manifests" / "data_manifest.json", {"sha256": manifest_hash, "files": len(tracked)})
    return statistics


def load_split(dataset: str, split: str) -> pd.DataFrame:
    path = data_dir() / "processed" / dataset / f"{split}.csv"
    return pd.read_csv(path, dtype={"example_id": str, "source_row_id": str})


def load_subset(dataset: str, seed: int, size: int) -> pd.DataFrame:
    pool = load_split(dataset, "train").set_index("example_id")
    manifest = pd.read_csv(data_dir() / "manifests" / f"subset_{dataset}_seed_{seed}.csv")
    ids = manifest.sort_values("rank")["example_id"].astype(str).tolist()
    if size > len(ids):
        raise ValueError(f"requested {size} examples from a pool of {len(ids)}")
    return pool.loc[ids[:size]].reset_index()


def data_manifest_hash() -> str:
    path = data_dir() / "manifests" / "data_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run preprocess")
    return json.loads(path.read_text(encoding="utf-8"))["sha256"]
