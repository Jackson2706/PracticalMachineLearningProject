"""Pytest setup: make ``src/`` importable; skip hardware-marked tests when absent."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ksent  # noqa: E402,F401  (imports xgboost before torch)
import pytest  # noqa: E402


def pytest_collection_modifyitems(config, items):
    from ksent.hardware import cuda_available, mps_available

    skip_cuda = pytest.mark.skip(reason="no CUDA device")
    skip_mps = pytest.mark.skip(reason="no Apple MPS device")
    for item in items:
        if "cuda" in item.keywords and not cuda_available():
            item.add_marker(skip_cuda)
        if "mps" in item.keywords and not mps_available():
            item.add_marker(skip_mps)
