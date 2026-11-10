from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from korean_sentiment.config import data_dir, dataset_config
from korean_sentiment.utils.io import atomic_write_csv, atomic_write_json, sha256_file


def _validate_parts(schema: str, parts: list[str], first: bool) -> bool:
    if schema == "nsmc":
        if first and [part.strip().lower() for part in parts] == ["id", "document", "label"]:
            return True
        return len(parts) >= 3 and parts[-1].strip() in {"0", "1"}
    if schema == "rating_text":
        return len(parts) >= 2 and parts[0].strip() in {"1", "2", "3", "4", "5"}
    if schema == "binary_text":
        return len(parts) >= 2 and parts[0].strip() in {"0", "1"}
    raise ValueError(f"unknown schema: {schema}")


def validate_download(path: Path, schema: str) -> int:
    rows = 0
    malformed = 0
    with path.open(encoding="utf-8-sig", errors="strict", newline=None) as handle:
        for line_number, line in enumerate(handle):
            line = line.rstrip("\r\n")
            if not line:
                malformed += 1
                continue
            parts = line.split("\t")
            if schema == "nsmc" and line_number == 0:
                if not _validate_parts(schema, parts, True):
                    raise ValueError(f"{path}: invalid NSMC header")
                continue
            rows += 1
            if not _validate_parts(schema, parts, False):
                malformed += 1
    if rows == 0 or malformed:
        raise ValueError(f"{path}: schema validation failed ({rows} rows, {malformed} malformed)")
    return rows


def _download(url: str, destination: Path, retries: int = 5, timeout: int = 45) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(retries):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "korean-sentiment-benchmark/0.1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                resumed = offset > 0 and response.status == 206
                mode = "ab" if resumed else "wb"
                with partial.open(mode) as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
            os.replace(partial, destination)
            return
        except (OSError, urllib.error.URLError, TimeoutError) as error:
            if attempt + 1 == retries:
                raise RuntimeError(f"download failed after {retries} attempts: {url}: {error}") from error
            time.sleep(min(30, 2**attempt))


def download_all(force: bool = False) -> list[dict[str, Any]]:
    config = dataset_config()["datasets"]
    manifest: list[dict[str, Any]] = []
    for dataset, dataset_spec in config.items():
        for source, file_spec in dataset_spec["files"].items():
            destination = data_dir() / "raw" / dataset / file_spec["name"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if force or not destination.exists():
                _download(file_spec["url"], destination)
            row_count = validate_download(destination, file_spec["schema"])
            destination.chmod(0o444)
            manifest.append(
                {
                    "dataset": dataset,
                    "source": source,
                    "file": str(destination),
                    "source_url": file_spec["url"],
                    "retrieved_at": datetime.fromtimestamp(destination.stat().st_mtime, timezone.utc).isoformat(),
                    "byte_size": destination.stat().st_size,
                    "row_count": row_count,
                    "sha256": sha256_file(destination),
                }
            )
    manifest.sort(key=lambda row: (row["dataset"], row["source"]))
    manifest_dir = data_dir() / "manifests"
    fields = ["dataset", "source", "file", "source_url", "retrieved_at", "byte_size", "row_count", "sha256"]
    atomic_write_csv(manifest_dir / "download_manifest.csv", manifest, fields)
    atomic_write_json(manifest_dir / "download_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(download_all(arguments.force), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
