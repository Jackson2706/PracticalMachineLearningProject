"""Pytest setup for the benchmark suite."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import xgboost  # noqa: E402,F401  (import before torch to avoid an OpenMP clash)
import pytest  # noqa: E402


def pytest_collection_modifyitems(config, items):
    import torch

    skip_cuda = pytest.mark.skip(reason="no CUDA device")
    skip_mps = pytest.mark.skip(reason="no Apple MPS device")
    for item in items:
        if "cuda" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_cuda)
        if "mps" in item.keywords and not torch.backends.mps.is_available():
            item.add_marker(skip_mps)


@pytest.fixture
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = tmp_path / "data"
    results = tmp_path / "results"
    monkeypatch.setenv("KS_DATA_DIR", str(data))
    monkeypatch.setenv("KS_RESULTS_DIR", str(results))
    for path in (
        data / "raw",
        data / "processed",
        data / "splits",
        data / "manifests",
        results / "plans",
        results / "run_records",
        results / "csv",
        results / "artifacts",
        results / "archives",
    ):
        path.mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def binary_texts():
    positive = ["정말 좋아요", "아주 재미있어요", "배송이 빨라요", "최고입니다"]
    negative = ["정말 싫어요", "아주 지루해요", "배송이 느려요", "최악입니다"]
    train = (positive + negative) * 2
    labels = [1] * 4 + [0] * 4 + [1] * 4 + [0] * 4
    validation = positive[:2] + negative[:2]
    validation_labels = [1, 1, 0, 0]
    return train, labels, validation, validation_labels
