"""Dataset statistics (Table 1), preprocessing diagnostics (Table A5, Fig. A1) and the
per-tokenizer neural ``max_seq_len``.

Usage:
  python -m ksent.data.stats --write-max-len --samples 10

Writes to ``results/aggregated/``:
  dataset_stats.json        cleaning counts, split stats, symbol coverage, max lengths
  diagnostics_oov.csv       vocab size and test OOV rate per dataset x tokenizer x size x seed
  diagnostics_truncation.csv truncation rate per dataset x tokenizer x split
"""
from __future__ import annotations

import argparse
import math
import unicodedata

import numpy as np
import pandas as pd

from ksent.config import global_cfg, main_tokenizer, path, processed_dir, write_max_seq_len
from ksent.data.split import SPLITS, load_split
from ksent.data.subsets import subset_ids
from ksent.features.tokenize import cached_tokens, surface
from ksent.features.vocab import Vocab
from ksent.utils import read_json, write_json

SYMBOLS: dict[str, str] = {"kieuk": "ㅋ", "yu": "ㅠ", "hieuh": "ㅎ"}


class SymbolDropError(RuntimeError):
    """Raised when sentiment symbols disappear during Kiwi tokenisation."""


def is_emoji(c: str) -> bool:
    cp = ord(c)
    return (0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF or 0x2B00 <= cp <= 0x2BFF) and \
        unicodedata.category(c) in ("So", "Sk")


def coverage_length(lengths: np.ndarray, share: float = 0.95, multiple: int = 8) -> int:
    """Smallest multiple of ``multiple`` such that >= ``share`` of ``lengths`` fit."""
    q = np.sort(lengths)[int(math.ceil(share * len(lengths))) - 1]
    return int(math.ceil(max(q, 1) / multiple) * multiple)


def _lengths(dataset: str, split: str, tok: str) -> np.ndarray:
    df = load_split(dataset, split)
    return np.array([len(t) for t in cached_tokens(dataset, df["review_id"], tok)])


def compute_max_seq_len() -> dict[str, dict]:
    """Per tokenizer: coverage length on the train pool of the dataset with the longest
    reviews (highest mean token count under that tokenizer)."""
    out = {}
    for tok in global_cfg()["tokenizers"]:
        means = {d: _lengths(d, "train_pool", tok).mean() for d in global_cfg()["datasets"]}
        longest = max(means, key=means.get)
        out[tok] = {"max_seq_len": coverage_length(_lengths(longest, "train_pool", tok)), "longest_dataset": longest,
                    "mean_lengths": {d: float(v) for d, v in means.items()}}
    return out


def truncation_table() -> pd.DataFrame:
    msl = global_cfg().get("max_seq_len") or {}
    rows = []
    for d in global_cfg()["datasets"]:
        for tok in global_cfg()["tokenizers"]:
            if not msl.get(tok):
                continue
            for s in SPLITS:
                lens = _lengths(d, s, tok)
                rows.append({"dataset": d, "tokenizer": tok, "split": s, "max_seq_len": msl[tok],
                             "truncation_rate": float((lens > msl[tok]).mean()), "len_mean": float(lens.mean())})
    return pd.DataFrame(rows)


def symbol_coverage(dataset: str) -> dict:
    """Share of reviews containing ㅋ/ㅠ/ㅎ/emoji and whether they survive Kiwi tokenisation."""
    frames = pd.concat([load_split(dataset, s) for s in SPLITS], ignore_index=True)
    toks = cached_tokens(dataset, frames["review_id"], "kiwi")
    res: dict = {"n_reviews": len(frames)}
    failures: list[str] = []
    any_mask = np.zeros(len(frames), dtype=bool)
    for name, ch in [*SYMBOLS.items(), ("emoji", None)]:
        has = frames["text"].map(lambda t: any(is_emoji(c) for c in t) if ch is None else ch in t).to_numpy()
        any_mask |= has
        lost = 0
        for i in np.flatnonzero(has):
            joined = "".join(surface(t) for t in toks[i])
            needed = {c for c in frames["text"].iat[i] if is_emoji(c)} if ch is None else {ch}
            if not needed <= set(joined):
                lost += 1
                if len(failures) < 5:
                    failures.append(f"{frames['text'].iat[i]!r} -> {toks[i]}")
        res[f"pct_{name}"] = float(has.mean() * 100)
        res[f"lost_{name}"] = lost
    res["pct_any_symbol"] = float(any_mask.mean() * 100)
    res["survival_ok"] = all(res[f"lost_{n}"] == 0 for n in [*SYMBOLS, "emoji"])
    res["failure_examples"] = failures
    return res


def oov_diagnostics() -> pd.DataFrame:
    """Neural vocabulary size and test OOV rates per dataset x tokenizer x size x seed."""
    rows = []
    for d in global_cfg()["datasets"]:
        test = load_split(d, "test")
        for tok in global_cfg()["tokenizers"]:
            test_docs = cached_tokens(d, test["review_id"], tok)
            for seed in global_cfg()["seeds"]:
                for n in global_cfg()["sizes"]:
                    vocab = Vocab.build(cached_tokens(d, subset_ids(d, seed, n), tok))
                    oov = [[t not in vocab.stoi for t in doc] for doc in test_docs]
                    n_tok = sum(len(o) for o in oov)
                    rows.append({"dataset": d, "tokenizer": tok, "size": n, "seed": seed, "vocab_size": len(vocab),
                                 "oov_token_rate": sum(map(sum, oov)) / max(n_tok, 1),
                                 "oov_review_rate": float(np.mean([any(o) for o in oov]))})
    return pd.DataFrame(rows)


def split_stats(dataset: str) -> dict:
    msl = (global_cfg().get("max_seq_len") or {}).get(main_tokenizer())
    out = {}
    for s in SPLITS:
        df = load_split(dataset, s)
        lens = _lengths(dataset, s, main_tokenizer())
        out[s] = {"n": int(len(df)), "pct_positive": float(df["label"].mean() * 100),
                  "tok_mean": float(lens.mean()), "tok_median": float(np.median(lens)),
                  "tok_p95": float(np.percentile(lens, 95)),
                  "trunc_pct": float((lens > msl).mean() * 100) if msl else None}
    return out


def cleaning_summary(dataset: str) -> dict:
    counts = read_json(processed_dir(dataset) / "cleaning_counts.json")
    keys = ["raw", "malformed", "neutral_dropped", "empty", "conflicting_dup", "exact_dup", "kiwi_empty",
            "cross_file_overlap", "clean"]
    total = {k: int(sum(c.get(k, 0) for c in counts.values())) for k in keys}
    total["per_file"] = counts
    return total


def print_samples(dataset: str, k: int) -> None:
    df = load_split(dataset, "train_pool").sample(k, random_state=0)
    toks = cached_tokens(dataset, df["review_id"], "kiwi")
    print(f"  --- {k} random reviews: raw -> normalized -> Kiwi tokens")
    for (_, r), t in zip(df.iterrows(), toks):
        print(f"   [{r['label']}] raw : {r['text_raw'][:110]}")
        print(f"       norm: {r['text'][:110]}")
        print(f"       kiwi: {' '.join(t[:30])}{' ...' if len(t) > 30 else ''}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-max-len", action="store_true")
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--skip-oov", action="store_true")
    args = ap.parse_args()
    agg = path("results") / "aggregated"
    stats: dict = {"tokenizer": main_tokenizer(), "datasets": {}}
    if args.write_max_len:
        msl = compute_max_seq_len()
        write_max_seq_len({t: v["max_seq_len"] for t, v in msl.items()})
        stats["max_seq_len_detail"] = msl
        for t, v in msl.items():
            print(f"max_seq_len[{t}] = {v['max_seq_len']} (longest dataset: {v['longest_dataset']})")
    stats["max_seq_len"] = global_cfg().get("max_seq_len")
    problems = []
    for ds in global_cfg()["datasets"]:
        sym = symbol_coverage(ds)
        stats["datasets"][ds] = {"cleaning": cleaning_summary(ds), "splits": split_stats(ds), "symbols": sym}
        c = stats["datasets"][ds]["cleaning"]
        print(f"[{ds}] raw={c['raw']:,} clean={c['clean']:,} empty={c['empty']} conflict={c['conflicting_dup']} "
              f"dup={c['exact_dup']} kiwi_empty={c['kiwi_empty']} rating3={c['neutral_dropped']} "
              f"xfile={c['cross_file_overlap']}")
        for s, v in stats["datasets"][ds]["splits"].items():
            print(f"    {s:10s} n={v['n']:5d} pos={v['pct_positive']:5.1f}% kiwi tok mean/med/p95="
                  f"{v['tok_mean']:.1f}/{v['tok_median']:.0f}/{v['tok_p95']:.0f} trunc={v['trunc_pct']}")
        print(f"    symbols: ㅋ {sym['pct_kieuk']:.1f}%  ㅠ {sym['pct_yu']:.1f}%  ㅎ {sym['pct_hieuh']:.1f}%  "
              f"emoji {sym['pct_emoji']:.2f}%  any {sym['pct_any_symbol']:.1f}%  survival_ok={sym['survival_ok']}")
        if not sym["survival_ok"]:
            problems.append((ds, sym))
        if args.samples:
            print_samples(ds, args.samples)
    agg.mkdir(parents=True, exist_ok=True)
    write_json(stats, agg / "dataset_stats.json")
    tt = truncation_table()
    tt.to_csv(agg / "diagnostics_truncation.csv", index=False)
    if len(tt):
        print(tt[tt.split == "train_pool"].pivot(index="dataset", columns="tokenizer", values="truncation_rate").round(4))
    if not args.skip_oov:
        oov = oov_diagnostics()
        oov.to_csv(agg / "diagnostics_oov.csv", index=False)
        print(oov[oov.tokenizer == "kiwi"].groupby(["dataset", "size"])[["vocab_size", "oov_token_rate",
                                                                       "oov_review_rate"]].mean().round(4))
    if problems:
        msg = "; ".join(f"{d}: lost={ {k: v for k, v in s.items() if k.startswith('lost_')} } e.g. {s['failure_examples'][:2]}"
                        for d, s in problems)
        raise SymbolDropError(f"Sentiment symbols dropped by Kiwi tokenisation: {msg}")


if __name__ == "__main__":
    main()
