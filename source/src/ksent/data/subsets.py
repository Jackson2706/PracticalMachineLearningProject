"""Nested stratified training subsets 500 ⊂ 1,000 ⊂ ... ⊂ 64,000 ⊂ full.

For every experiment seed (and the tuning seed) one stratified random ordering of
the train pool is drawn, followed by one of ``train_extra``; each training size is a
prefix of the concatenated ordering, so the subsets are nested and every prefix keeps
the class ratio (within one example). Sizes up to ``splits.train_pool`` (8,000) only
use the train pool, exactly as before ``train_extra`` existed.

Usage: ``python -m ksent.data.subsets`` -> data/processed/{dataset}/subsets_seed{seed}.json
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ksent.config import global_cfg, processed_dir, resolve_sizes
from ksent.data.split import EXTRA, load_split
from ksent.utils import read_json, write_json


def stratified_order(ids: np.ndarray, labels: np.ndarray, seed: int | list[int]) -> list[str]:
    """Return a stratified random ordering of ``ids``.

    Each item of class ``c`` (with ``n_c`` members) at within-class random rank ``r``
    gets the key ``(r + u) / n_c`` with ``u ~ U(0, 1)``; sorting by key interleaves the
    classes proportionally, so any prefix of length ``k`` holds ≈ ``k · n_c / n`` items
    of class ``c``.
    """
    rng = np.random.default_rng(seed)
    keys = np.empty(len(ids), dtype=float)
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        perm = rng.permutation(len(idx))
        keys[idx[perm]] = (np.arange(len(idx)) + rng.random(len(idx))) / len(idx)
    order = np.argsort(keys, kind="stable")
    return ids[order].tolist()


def make_subsets(dataset: str, seed: int) -> dict:
    """Create and save the nested subsets of ``dataset`` for ``seed``."""
    pool = load_split(dataset, "train_pool").sort_values("review_id")
    order = stratified_order(pool["review_id"].to_numpy(), pool["label"].to_numpy(), seed)
    extra = load_split(dataset, EXTRA).sort_values("review_id")
    if len(extra):  # separate stream so the train-pool ordering is unchanged
        order += stratified_order(extra["review_id"].to_numpy(), extra["label"].to_numpy(), [seed, 1])
    sizes = sorted(set(resolve_sizes(dataset)) | {int(global_cfg()["tuning_size"])})
    obj = {"dataset": dataset, "seed": seed, "sizes": sizes, "order": order}
    write_json(obj, processed_dir(dataset) / f"subsets_seed{seed}.json")
    return obj


def subset_ids(dataset: str, seed: int, size: int) -> list[str]:
    """IDs of the training subset of ``size`` for ``seed`` (a prefix of the ordering)."""
    obj = read_json(processed_dir(dataset) / f"subsets_seed{seed}.json")
    if size > len(obj["order"]):
        raise ValueError(f"size {size} > pool {len(obj['order'])}")
    return obj["order"][:size]


def load_subset(dataset: str, seed: int, size: int) -> pd.DataFrame:
    """Training subset as a frame, in the subset's (stratified random) order."""
    ids = subset_ids(dataset, seed, size)
    pool = load_split(dataset, "train_pool")
    if size > len(pool):
        pool = pd.concat([pool, load_split(dataset, EXTRA)], ignore_index=True)
    return pool.set_index("review_id").loc[ids].reset_index()


def all_seeds() -> list[int]:
    """Experiment seeds plus the tuning seed."""
    return [int(global_cfg()["tuning_seed"])] + [int(s) for s in global_cfg()["seeds"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    for ds in global_cfg()["datasets"]:
        for seed in all_seeds():
            obj = make_subsets(ds, seed)
            print(f"[{ds}] seed={seed}: {len(obj['order'])} ordered ids")


if __name__ == "__main__":
    main()
