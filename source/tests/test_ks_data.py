from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from korean_sentiment.data.download import validate_download
from korean_sentiment.data.prepare import (
    clean_frame,
    make_fixed_splits,
    normalize_text,
    parse_file,
    size_schedule,
    stratified_order,
    validate_nested_manifest,
    validate_splits,
)
from korean_sentiment.features.text import UnicodeWordTokenizer, Vocabulary
from korean_sentiment.models.pipelines import TfidfPipeline


def test_ks_nsmc_parser_and_schema(tmp_path: Path):
    path = tmp_path / "ratings.txt"
    path.write_text("id\tdocument\tlabel\n1\t좋아요 :)\t1\n2\t싫어요\t0\n", encoding="utf-8")
    assert validate_download(path, "nsmc") == 2
    frame, counts = parse_file("nsmc", "train", path)
    assert frame["label"].tolist() == [1, 0]
    assert counts["malformed_rows"] == 0


def test_ks_shopping_rating_mapping_and_neutral_drop(tmp_path: Path):
    path = tmp_path / "shopping.txt"
    path.write_text("1\t나쁨\n2\t별로\n3\t보통\n4\t좋음\n5\t최고\n", encoding="utf-8")
    assert validate_download(path, "rating_text") == 5
    frame, counts = parse_file("shopping", "all", path)
    assert frame["label"].tolist() == [0, 0, 1, 1]
    assert counts["neutral_rating_rows"] == 1


def test_ks_steam_binary_mapping(tmp_path: Path):
    path = tmp_path / "steam.txt"
    path.write_text("1\t추천함\n0\t추천 안 함\n", encoding="utf-8")
    assert validate_download(path, "binary_text") == 2
    frame, _ = parse_file("steam", "all", path)
    assert frame["label"].tolist() == [1, 0]


def _raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["x"] * 7,
            "source": ["all"] * 7,
            "source_row_id": [f"all:{i}" for i in range(7)],
            "original_id": [str(i) for i in range(7)],
            "text_raw": [" 좋다  :) ", "좋다 :) ", "충돌", " 충돌 ", "", "싫다!!!", "싫다!!!"],
            "label": [1, 1, 0, 1, 0, 0, 0],
        }
    )


def test_ks_cleaning_preserves_marks_and_removes_conflicts_and_duplicates():
    cleaned, report = clean_frame(_raw_frame())
    assert cleaned["text"].tolist() == ["좋다 :)", "싫다!!!"]
    assert report["empty_removed"] == 1
    assert report["conflicting_occurrences_removed"] == 2
    assert report["duplicate_pairs_removed"] == 2
    assert normalize_text("안  좋아요ㅠㅠ!!!") == "안 좋아요ㅠㅠ!!!"


def _clean_split_frame(n: int = 100) -> pd.DataFrame:
    labels = [index % 2 for index in range(n)]
    return pd.DataFrame(
        {
            "dataset": "shopping",
            "source": "all",
            "source_row_id": [f"all:{i}" for i in range(n)],
            "original_id": [str(i) for i in range(n)],
            "text_raw": [f"문장 {i}" for i in range(n)],
            "text": [f"문장 {i}" for i in range(n)],
            "normalized_text_key": [f"문장 {i}" for i in range(n)],
            "label": labels,
            "text_hash": [f"hash-{i}" for i in range(n)],
            "example_id": [f"shopping:all:{i}" for i in range(n)],
        }
    )


def test_ks_fixed_split_is_stratified_disjoint_and_stable():
    frame = _clean_split_frame()
    first, _ = make_fixed_splits("shopping", {"all": frame}, 2026)
    second, _ = make_fixed_splits("shopping", {"all": frame}, 2026)
    assert [len(first[name]) for name in ("train", "validation", "test")] == [80, 10, 10]
    validate_splits(first)
    for name in first:
        assert first[name]["example_id"].tolist() == second[name]["example_id"].tolist()
        assert first[name]["label"].mean() == pytest.approx(0.5)


def test_ks_leakage_detector_rejects_text_overlap():
    frame = _clean_split_frame(4)
    train = frame.iloc[:2].copy()
    test = frame.iloc[[1, 2]].copy()
    test["example_id"] = ["new-1", "new-2"]
    with pytest.raises(AssertionError, match="text leakage"):
        validate_splits({"train": train, "test": test})


@pytest.mark.parametrize(
    ("full", "expected"),
    [
        (135000, [200, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 135000]),
        (160000, [200, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 160000]),
        (80000, [200, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 80000]),
    ],
)
def test_ks_dynamic_size_schedule(full, expected):
    assert size_schedule(full) == expected


def test_ks_nested_order_is_deterministic_and_balanced():
    frame = _clean_split_frame(100)
    first = stratified_order(frame, 13)
    assert first == stratified_order(frame, 13)
    validate_nested_manifest(frame, first, [20, 50, 100])
    labels = frame.set_index("example_id").loc[first[:51], "label"]
    assert abs(int(labels.sum()) - 25.5) <= 1


def test_ks_vocabulary_and_tfidf_fit_training_only():
    tokenizer = UnicodeWordTokenizer()
    vocab = Vocabulary.fit([tokenizer("훈련 전용")])
    assert "검증전용" not in vocab.stoi
    pipeline = TfidfPipeline("tfidf_lr", {"C": 1.0, "solver": "liblinear", "max_iter": 100}, 1)
    pipeline.fit(["훈련 전용", "나쁜 훈련"], [1, 0])
    assert all("검증전용" not in value for value in pipeline.vectorizer.get_feature_names_out())
