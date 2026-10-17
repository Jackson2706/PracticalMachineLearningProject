"""Download the raw review files and print a format check.

Usage: ``python -m ksent.data.download [--force]``
"""
from __future__ import annotations

import argparse
import collections
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from ksent.config import data_cfg, path


def raw_file(dataset: str, key: str) -> Path:
    """Local path of raw file ``key`` (e.g. ``train``/``test``/``all``) of ``dataset``."""
    name = data_cfg()["datasets"][dataset]["files"][key]["name"]
    return path("data_raw") / dataset / name


def _fetch(url: str, dst: Path, retries: int = 3) -> None:
    """Download ``url`` to ``dst`` with urllib, retrying, then falling back to curl."""
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dst)
            return
        except Exception as e:  # network hiccup / TLS reset
            print(f"    urllib attempt {attempt + 1} failed: {e}")
            time.sleep(2 * (attempt + 1))
    if shutil.which("curl"):
        subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", str(dst), url], check=True)
        return
    raise RuntimeError(f"could not download {url}")


def download(dataset: str, force: bool = False) -> list[Path]:
    """Download every raw file of ``dataset`` (skips existing files unless ``force``)."""
    out = []
    for key, spec in data_cfg()["datasets"][dataset]["files"].items():
        dst = raw_file(dataset, key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if force or not dst.exists() or dst.stat().st_size == 0:
            print(f"  downloading {spec['url']} -> {dst}")
            tmp = dst.with_suffix(".part")
            _fetch(spec["url"], tmp)
            tmp.replace(dst)
        out.append(dst)
    return out


def inspect_file(p: Path, n_lines: int = 3) -> dict[int, int]:
    """Print the first lines of ``p`` and return the distribution of tab-field counts."""
    counts: collections.Counter[int] = collections.Counter()
    with open(p, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if i < n_lines:
                print(f"    {i}: {line[:100]!r}")
            counts[line.count("\t") + 1] += 1
    print(f"    lines={sum(counts.values()):,}  field-count distribution={dict(counts)}")
    return dict(counts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for ds in data_cfg()["datasets"]:
        print(f"[{ds}]")
        for p in download(ds, force=args.force):
            print(f"  {p.name}")
            inspect_file(p)


if __name__ == "__main__":
    main()
