#!/usr/bin/env bash
# Full reproduction: 01 -> 08 (run 00_setup_env.sh once beforehand).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
for s in 01_prepare_data 02_smoke_test 03_pilot 04_tune 05_run_all 06_run_ablation 07_analyze 08_build_submission; do
  bash "$DIR/$s.sh"
done
source "$DIR/_common.sh"
python -m pytest -q tests
