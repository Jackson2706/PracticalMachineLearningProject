"""Tokenizers shared by all nine methods. All operate on *normalized* text.

=========== ================================== ===========================================
name        TF-IDF features                    neural tokens
=========== ================================== ===========================================
kiwi (main) morpheme unigrams + bigrams        morphemes as ``form/TAG`` (e.g. 좋/VA, 안/MAG)
whitespace  eojeol unigrams + bigrams          eojeols
char        character 2-4-grams (on the text)  single characters (no whitespace)
=========== ================================== ===========================================

Kiwi keeps every morpheme, including particles and endings (negation 안/못/-지 않-,
contrast -는데, tone -네); typo correction is disabled. Tokenisation is deterministic,
so each dataset is tokenised once per tokenizer and cached in
``data/tokenized/{dataset}/{tokenizer}.parquet``.

Usage: ``python -m ksent.features.tokenize [--tokenizers kiwi whitespace char]``
"""
from __future__ import annotations

import argparse
import functools
import time
from collections.abc import Callable, Iterable

import pandas as pd

from ksent.config import global_cfg, main_tokenizer, path
from ksent.data.normalize import normalize
from ksent.utils import write_json

Tokens = list[str]
TOKENIZERS: tuple[str, ...] = ("kiwi", "whitespace", "char")


@functools.lru_cache(maxsize=1)
def _kiwi():
    """One shared Kiwi instance per process (typo correction off; batch calls use
    ``n_threads`` workers, single strings run on the calling thread)."""
    from kiwipiepy import Kiwi

    from ksent.hardware import configured_threads

    return Kiwi(num_workers=configured_threads(), typos=None)


def _kiwi_forms(tokens) -> Tokens:
    return [f"{t.form}/{t.tag}" for t in tokens]


def _char_tokens(text: str) -> Tokens:
    return [c for c in text if not c.isspace()]


def get_tokenizer(name: str | None = None) -> Callable[[str], Tokens]:
    """``normalized str -> tokens`` for ``name``."""
    name = name or main_tokenizer()
    if name == "kiwi":
        kiwi = _kiwi()
        return lambda text: _kiwi_forms(kiwi.tokenize(text))
    if name == "whitespace":
        return str.split
    if name == "char":
        return _char_tokens
    raise ValueError(f"unknown tokenizer {name!r}")


def tokenize_many(texts: Iterable[str], name: str | None = None) -> list[Tokens]:
    """Tokenise many normalized texts (Kiwi uses its internal thread pool)."""
    name = name or main_tokenizer()
    texts = list(texts)
    if name == "kiwi":
        return [_kiwi_forms(toks) for toks in _kiwi().tokenize(texts)]
    tok = get_tokenizer(name)
    return [tok(t) for t in texts]


def raw_to_tokens(raw: str, name: str | None = None) -> Tokens:
    """End-to-end preprocessing of one raw review: normalize -> tokenize."""
    return get_tokenizer(name)(normalize(raw))


def tfidf_uses_text(name: str) -> bool:
    """The ``char`` TF-IDF analyser works on the normalized string, not on tokens."""
    return name == "char"


def surface(token: str) -> str:
    """Surface form of a token (``좋/VA`` -> ``좋``)."""
    return token.rsplit("/", 1)[0] if "/" in token[1:] else token


# ----------------------------------------------------------------------------- cache
def cache_path(dataset: str, name: str):
    return path("data_tokenized") / dataset / f"{name}.parquet"


def build_cache(dataset: str, name: str, force: bool = False) -> dict:
    """Tokenise every split of ``dataset`` once; return timing info."""
    from ksent.data.split import SPLITS, load_split

    p = cache_path(dataset, name)
    if p.exists() and not force:
        return {"cached": True}
    df = pd.concat([load_split(dataset, s)[["review_id", "text"]] for s in SPLITS], ignore_index=True)
    t0 = time.perf_counter()
    df["tokens"] = tokenize_many(df["text"], name)
    elapsed = time.perf_counter() - t0
    empty = int((df["tokens"].map(len) == 0).sum())
    p.parent.mkdir(parents=True, exist_ok=True)
    df[["review_id", "tokens"]].to_parquet(p, index=False)
    info = {"tokenizer": name, "n_docs": len(df), "seconds": elapsed, "docs_per_s": len(df) / max(elapsed, 1e-9),
            "n_empty": empty}
    write_json(info, p.with_suffix(".timing.json"))
    return info


@functools.lru_cache(maxsize=16)
def _load_cache(dataset: str, name: str) -> dict[str, Tokens]:
    df = pd.read_parquet(cache_path(dataset, name))
    return {rid: list(toks) for rid, toks in zip(df["review_id"], df["tokens"])}


def cached_tokens(dataset: str, ids: Iterable[str], name: str | None = None) -> list[Tokens]:
    """Look up cached tokens for ``ids`` (builds the cache if missing)."""
    name = name or main_tokenizer()
    if not cache_path(dataset, name).exists():
        build_cache(dataset, name)
    cache = _load_cache(dataset, name)
    return [cache[i] for i in ids]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizers", nargs="*", default=list(global_cfg()["tokenizers"]))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for ds in global_cfg()["datasets"]:
        for name in args.tokenizers:
            info = build_cache(ds, name, force=args.force)
            print(f"[{ds}] {name}: {info}")


if __name__ == "__main__":
    main()
