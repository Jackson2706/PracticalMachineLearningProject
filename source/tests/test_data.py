"""Data integrity: no overlap, stratification within ±1%, nesting, determinism."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from ksent.config import global_cfg, processed_dir
from ksent.data.clean import clean_frame
from ksent.data.normalize import normalize
from ksent.data.subsets import stratified_order

DATASETS = global_cfg()["datasets"]
SIZES = sorted(set(global_cfg()["sizes"]))
SEEDS = [int(global_cfg()["tuning_seed"])] + [int(s) for s in global_cfg()["seeds"]]


def _have(ds: str) -> bool:
    return (processed_dir(ds) / "test.parquet").exists()


# ---------------------------------------------------------------- unit tests (always run)
def test_normalize_rules_in_order() -> None:
    assert normalize("  재밌다   ㅋㅋㅋㅋㅋ!!!!!  ") == "재밌다 ㅋㅋ!!"            # cap repeats, whitespace
    assert normalize("ㅠㅠㅠ 최악......") == "ㅠㅠ 최악.."
    assert normalize("&lt;3 &amp; GOOD Movie") == "<3 & good movie"           # html.unescape, Latin lowercase
    assert normalize("좋\u200b아\ufeff요\x07") == "좋아요"                     # zero-width + control chars
    assert normalize("10점 만점에 1점") == "10점 만점에 1점"                     # digits kept
    assert normalize("ㅋ") == "\u314b"                                          # compatibility jamo kept (NFC, not NFKC)
    assert normalize("\u1100\u1161") == "가"                                    # NFC composes conjoining jamo
    assert normalize("😀😀😀 ㅅㅂ") == "😀😀 ㅅㅂ"                                # emoji, profanity kept
    assert normalize(None) == ""


def test_clean_frame_rules() -> None:
    df = pd.DataFrame({"text_raw": ["좋아", "좋아 ", "별로", "별로", "  ", "굿", None, "\u200b"],
                       "label": [1, 1, 0, 1, 1, 1, 0, 1]})
    out, c = clean_frame(df)
    assert c == {"parsed": 8, "empty": 3, "conflicting_dup": 2, "exact_dup": 1, "kiwi_empty": 0, "clean": 2}
    assert sorted(out.text) == ["굿", "좋아"]
    assert "text_raw" in out


def test_stratified_order_synthetic() -> None:
    rng = np.random.default_rng(0)
    labels = (rng.random(8000) < 0.37).astype(int)
    ids = np.array([f"r{i:05d}" for i in range(8000)])
    order = stratified_order(ids, labels, seed=1)
    assert sorted(order) == sorted(ids.tolist())
    lab = dict(zip(ids, labels))
    for n in SIZES:
        frac = np.mean([lab[i] for i in order[:n]])
        assert abs(frac - labels.mean()) <= 0.01
    assert order == stratified_order(ids, labels, seed=1)       # deterministic
    assert order != stratified_order(ids, labels, seed=2)       # seed matters


# ---------------------------------------------------------------- real data (skipped if not prepared)
@pytest.mark.parametrize("ds", DATASETS)
def test_splits_disjoint_and_sized(ds: str) -> None:
    if not _have(ds):
        pytest.skip("run scripts/01_prepare_data.sh first")
    sp = {s: pd.read_parquet(processed_dir(ds) / f"{s}.parquet") for s in ("train_pool", "val", "test")}
    assert [len(sp[s]) for s in sp] == [8000, 2000, 2000]
    names = list(sp)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not set(sp[a].text) & set(sp[b].text), f"text overlap {a}/{b}"
            assert not set(sp[a].review_id) & set(sp[b].review_id)
    for df in sp.values():
        assert df.review_id.is_unique and df.text.is_unique
        assert (df.text.str.len() > 0).all()
        assert (df.text == df.text_raw.map(normalize)).all()


@pytest.mark.parametrize("ds", DATASETS)
def test_split_stratification(ds: str) -> None:
    if not _have(ds):
        pytest.skip("run scripts/01_prepare_data.sh first")
    pos = [pd.read_parquet(processed_dir(ds) / f"{s}.parquet").label.mean() for s in ("train_pool", "val", "test")]
    if ds != "nsmc":  # NSMC test comes from a different file
        assert max(pos) - min(pos) <= 0.01
    assert abs(pos[0] - pos[1]) <= 0.01


@pytest.mark.parametrize("ds", DATASETS)
def test_subsets_nested_and_stratified(ds: str) -> None:
    if not _have(ds):
        pytest.skip("run scripts/01_prepare_data.sh first")
    from ksent.data.subsets import subset_ids

    pool = pd.read_parquet(processed_dir(ds) / "train_pool.parquet").set_index("review_id")
    p_all = pool.label.mean()
    for seed in SEEDS:
        prev: set[str] = set()
        for n in SIZES:
            ids = subset_ids(ds, seed, n)
            assert len(ids) == len(set(ids)) == n
            assert prev <= set(ids), "subsets must be nested"
            assert abs(pool.loc[ids].label.mean() - p_all) <= 0.01
            prev = set(ids)
        assert prev == set(pool.index)  # largest size uses the full pool
    a, b = subset_ids(ds, SEEDS[1], 500), subset_ids(ds, SEEDS[2], 500)
    assert a != b


@pytest.mark.parametrize("ds", DATASETS)
def test_split_determinism(ds: str) -> None:
    """Re-running the split step reproduces byte-identical splits."""
    if not _have(ds):
        pytest.skip("run scripts/01_prepare_data.sh first")
    from ksent.data.split import make_splits

    def digest() -> str:
        h = hashlib.sha256()
        for s in ("train_pool", "val", "test"):
            df = pd.read_parquet(processed_dir(ds) / f"{s}.parquet")
            h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
        return h.hexdigest()

    before = digest()
    make_splits(ds)
    assert digest() == before


# ---------------------------------------------------------------- tokenization (real data)
def test_symbol_survival() -> None:
    if not _have("nsmc"):
        pytest.skip("run scripts/01_prepare_data.sh first")
    from ksent.data.stats import symbol_coverage

    for ds in DATASETS:
        res = symbol_coverage(ds)
        assert res["survival_ok"], (ds, res["failure_examples"])
        assert res["pct_any_symbol"] > 0


def test_kiwi_tokens_have_tags_and_keep_symbols() -> None:
    from ksent.features.tokenize import get_tokenizer

    toks = get_tokenizer("kiwi")(normalize("영화 안 봤는데 좋네요 ㅋㅋㅋㅋ 😀"))
    assert "안/MAG" in toks and "좋/VA" in toks and "는데/EC" in toks
    assert any(t.startswith("ㅋㅋ/") for t in toks) and any(t.startswith("😀/") for t in toks)


def test_an_mag_in_tfidf_vocab_nsmc_2000() -> None:
    """sklearn's default token_pattern would drop 1-character tokens such as 안/좋/못."""
    if not _have("nsmc"):
        pytest.skip("run scripts/01_prepare_data.sh first")
    from ksent.data.subsets import subset_ids
    from ksent.features.tfidf import make_vectorizer
    from ksent.features.tokenize import cached_tokens

    docs = cached_tokens("nsmc", subset_ids("nsmc", 0, 2000), "kiwi")
    vec = make_vectorizer("kiwi").fit(docs)
    assert "안/MAG" in vec.vocabulary_
    assert any(" " in k for k in vec.vocabulary_)  # bigrams present
