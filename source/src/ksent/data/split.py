"""Fixed stratified train-pool / validation / test splits (data seed 2026).

Usage: ``python -m ksent.data.split``  ->  data/processed/{dataset}/{train_pool,val,test}.parquet
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ksent.config import data_cfg, global_cfg, processed_dir
from ksent.data.clean import load_clean
from ksent.utils import write_json

SPLITS: tuple[str, ...] = ("train_pool", "val", "test")


def _stratified_take(df: pd.DataFrame, n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified sample of exactly ``n`` rows; returns (taken, rest)."""
    if n > len(df):
        raise ValueError(f"need {n} rows, only {len(df)} available")
    if n == len(df):
        return df, df.iloc[0:0]
    taken, rest = train_test_split(df, train_size=n, stratify=df["label"], random_state=seed)
    return taken, rest


def make_splits(dataset: str, seed: int | None = None) -> dict[str, pd.DataFrame]:
    """Build the three fixed splits of ``dataset`` (deterministic given ``seed``)."""
    seed = int(global_cfg()["data_seed"] if seed is None else seed)
    sizes = data_cfg()["splits"]
    frames, counts = load_clean(dataset)
    for key, df in frames.items():
        df["review_id"] = [f"{dataset}-{key}-{r:07d}" for r in df["raw_row"]]
    if data_cfg()["datasets"][dataset]["test_source"] == "official_test":
        pool_val, _ = _stratified_take(frames["train"], sizes["train_pool"] + sizes["val"], seed)
        pool, val = _stratified_take(pool_val, sizes["train_pool"], seed)
        test, _ = _stratified_take(frames["test"], sizes["test"], seed)
    else:
        total = sizes["train_pool"] + sizes["val"] + sizes["test"]
        sample, _ = _stratified_take(frames["all"], total, seed)
        pool, rest = _stratified_take(sample, sizes["train_pool"], seed)
        val, test = _stratified_take(rest, sizes["val"], seed)
    out = {}
    for name, df in zip(SPLITS, (pool, val, test)):
        keep = ["review_id", "text_raw", "text", "label"] + [c for c in ("rating", "orig_id") if c in df]
        out[name] = df[keep].sort_values("review_id").reset_index(drop=True)
    check_disjoint(out)
    out_dir = processed_dir(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
    write_json(counts, out_dir / "cleaning_counts.json")
    # fixed latency reviews (same 200 for every method)
    n_lat = int(global_cfg()["latency"]["n_reviews"])
    rng = np.random.default_rng(seed)
    lat_ids = sorted(rng.choice(out["test"]["review_id"].to_numpy(), size=n_lat, replace=False).tolist())
    write_json({"seed": seed, "review_ids": lat_ids}, out_dir / "latency_ids.json")
    return out


def check_disjoint(splits: dict[str, pd.DataFrame]) -> None:
    """Assert no review id or *normalized* text is shared across splits."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ids = set(splits[a]["review_id"]) & set(splits[b]["review_id"])
            txt = set(splits[a]["text"]) & set(splits[b]["text"])
            assert not ids, f"id overlap {a}/{b}: {len(ids)}"
            assert not txt, f"text overlap {a}/{b}: {len(txt)}"


def load_split(dataset: str, split: str) -> pd.DataFrame:
    """Load one processed split."""
    return pd.read_parquet(processed_dir(dataset) / f"{split}.parquet")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()
    for ds in args.datasets or global_cfg()["datasets"]:
        out = make_splits(ds)
        desc = ", ".join(f"{k}={len(v)} ({v['label'].mean():.1%} pos)" for k, v in out.items())
        print(f"[{ds}] {desc}")


if __name__ == "__main__":
    main()
