"""Parse raw files and clean them (on normalized text, see :mod:`ksent.data.normalize`).

Cleaning steps, each logged for Table 1:
  1. drop empty texts (after normalization);
  2. drop *all* copies of texts that appear with conflicting labels;
  3. keep one copy of exact duplicates;
  4. drop reviews that yield zero Kiwi tokens (expected ~0).
Ratings, ids and flags are kept only as metadata and never reach the models.
"""
from __future__ import annotations

import pandas as pd

from ksent.config import data_cfg
from ksent.data.download import raw_file
from ksent.data.normalize import normalize


def parse_raw(dataset: str, key: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parse one raw file into ``(raw_row, source, text, label[, rating])``.

    Returns the frame and counts of malformed lines and neutral ratings dropped.
    """
    spec = data_cfg()["datasets"][dataset]
    cols = spec["format"]["columns"]
    rows: list[dict] = []
    counts = {"raw": 0, "malformed": 0, "neutral_dropped": 0}
    rating_map = data_cfg()["rating_map"]
    with open(raw_file(dataset, key), encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0 and spec["format"]["header"]:
                continue
            counts["raw"] += 1
            line = line.rstrip("\n").rstrip("\r")
            parts = line.split("\t")
            if len(parts) < len(cols):
                counts["malformed"] += 1
                continue
            if cols == ["id", "text", "label"]:
                # label is the last field, id the first; text may in theory contain tabs
                rid, text, lab = parts[0], "\t".join(parts[1:-1]), parts[-1]
                rec = {"orig_id": rid, "text": text, "label_raw": lab}
            elif cols == ["rating", "text"]:
                rec = {"rating_raw": parts[0], "text": "\t".join(parts[1:])}
            elif cols == ["label", "text"]:
                rec = {"label_raw": parts[0], "text": "\t".join(parts[1:])}
            else:  # pragma: no cover
                raise ValueError(f"unsupported column layout {cols}")
            try:
                if spec["label_rule"] == "rating":
                    rating = int(rec.pop("rating_raw").strip())
                    if rating in rating_map["drop"]:
                        counts["neutral_dropped"] += 1
                        continue
                    if rating in rating_map["negative"]:
                        rec["label"] = 0
                    elif rating in rating_map["positive"]:
                        rec["label"] = 1
                    else:
                        raise ValueError(rating)
                    rec["rating"] = rating
                else:
                    lab = int(rec.pop("label_raw").strip())
                    if lab not in (0, 1):
                        raise ValueError(lab)
                    rec["label"] = lab
            except ValueError:
                counts["malformed"] += 1
                continue
            rec["text_raw"] = rec.pop("text")
            rec["raw_row"] = i
            rec["source"] = key
            rows.append(rec)
    return pd.DataFrame(rows), counts


def clean_frame(df: pd.DataFrame, kiwi_check: bool = True) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normalize ``text_raw`` into ``text`` and apply the removal steps; return frame and counts."""
    counts: dict[str, int] = {"parsed": len(df)}
    df = df.copy()
    if "text_raw" not in df:
        df["text_raw"] = df["text"]
    max_rep = int(data_cfg()["normalization"]["max_repeat"])
    df["text"] = df["text_raw"].map(lambda t: normalize(t, max_rep))
    nonempty = df["text"].str.len() > 0
    counts["empty"] = int((~nonempty).sum())
    df = df[nonempty]
    n_labels = df.groupby("text")["label"].transform("nunique")
    conflicting = n_labels > 1
    counts["conflicting_dup"] = int(conflicting.sum())
    df = df[~conflicting]
    dup = df.duplicated(subset=["text"], keep="first")
    counts["exact_dup"] = int(dup.sum())
    df = df[~dup].reset_index(drop=True)
    counts["kiwi_empty"] = 0
    if kiwi_check and data_cfg()["cleaning"].get("drop_kiwi_empty", True):
        from ksent.features.tokenize import tokenize_many

        n_tok = pd.Series([len(t) for t in tokenize_many(df["text"].tolist(), "kiwi")], index=df.index)
        counts["kiwi_empty"] = int((n_tok == 0).sum())
        df = df[n_tok > 0].reset_index(drop=True)
    counts["clean"] = len(df)
    return df, counts


def load_clean(dataset: str) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, int]]]:
    """Parse and clean every raw file of ``dataset``.

    Returns ``{file_key: frame}`` and ``{file_key: counts}``. For NSMC the official
    test file is cleaned separately and test texts that also occur in the cleaned
    train file are removed (``cross_file_overlap``) to guarantee disjoint splits.
    """
    frames, counts = {}, {}
    for key in data_cfg()["datasets"][dataset]["files"]:
        raw, c_raw = parse_raw(dataset, key)
        if data_cfg()["datasets"][dataset]["label_rule"] == "rating":
            print(f"  [{dataset}] reviews with rating 3 (dropped): {c_raw['neutral_dropped']}")
        clean, c_clean = clean_frame(raw)
        frames[key], counts[key] = clean, {**c_raw, **c_clean}
    if "test" in frames and "train" in frames:
        overlap = frames["test"]["text"].isin(set(frames["train"]["text"]))
        counts["test"]["cross_file_overlap"] = int(overlap.sum())
        frames["test"] = frames["test"][~overlap].reset_index(drop=True)
        counts["test"]["clean"] = len(frames["test"])
    return frames, counts
