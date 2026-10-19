#!/usr/bin/env bash
# Full reproduction: 01 -> 08 (run 00_setup_env.sh once beforehand).
# Quality stages run in parallel; pilot, tuning and the cost pass run exclusively.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
bash "$DIR/01_prepare_data.sh"
bash "$DIR/02_smoke_test.sh"
bash "$DIR/03_pilot.sh"
bash "$DIR/04_tune.sh" --mode "${KSENT_TUNING_MODE:-cost}"
if [ "${KSENT_TUNING_MODE:-cost}" = quality ]; then bash "$DIR/04_tune.sh" --mode cost; fi
bash "$DIR/05_run_all.sh" --mode quality
bash "$DIR/06_run_ablation.sh" --mode quality
bash "$DIR/06b_cost_pass.sh"
bash "$DIR/07_analyze.sh"
bash "$DIR/08_build_submission.sh"
source "$DIR/_common.sh"
python -m pytest -q tests
