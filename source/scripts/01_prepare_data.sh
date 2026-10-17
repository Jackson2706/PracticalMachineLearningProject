#!/usr/bin/env bash
# download -> normalize + clean -> fixed splits -> tokenize (kiwi, whitespace, char; cached)
# -> nested subsets -> Table 1 stats + diagnostics (max_seq_len, truncation, OOV, symbols) -> data tests
set -euo pipefail
source "$(dirname "$0")/_common.sh"

python -m ksent.data.download            # prints first lines + tab-field counts per file
python -m ksent.data.split               # normalization, cleaning, stratified 8000/2000/2000 (seed 2026)
python -m ksent.features.tokenize        # data/tokenized/{dataset}/{tokenizer}.parquet
python -m ksent.data.subsets             # nested subsets for seed 0 (tuning) + 5 experiment seeds
python -m ksent.data.stats --write-max-len --samples 10   # fails loudly if symbols are dropped
python -m pytest -q tests/test_data.py
