import importlib.util
import os
from pathlib import Path


def test_storage_sampling_deduplicates_hardlinks_and_excludes_archives(tmp_path):
    spec = importlib.util.spec_from_file_location("storage_monitor", Path(__file__).resolve().parents[2] / "scripts/server_storage_monitor.py")
    sampler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sampler)
    model = tmp_path / "model.pt"
    model.write_bytes(b"x" * 1024)
    os.link(model, tmp_path / "same_model.pt")
    archives = tmp_path / "archives"
    archives.mkdir()
    (archives / "old.tar").write_bytes(b"x" * 8192)
    assert sampler.allocated_bytes([tmp_path]) == model.stat().st_blocks * 512
