#!/usr/bin/env bash
# 675 main fits (3 x 9 x 5 sizes x 5 seeds, kiwi) through the scheduler; resumable.
#   --mode quality (default): parallel, predictive metrics + predictions -> results/raw/
#   --mode cost             : exclusive, costs -> results/cost/raw/ (Stage G uses --sizes 8000)
# Usage: bash scripts/05_run_all.sh [--mode quality|cost] [--dataset ...] [--method ...] [--sizes ...] [--seeds ...]
# Long run:  nohup bash scripts/05_run_all.sh --mode quality > logs/run_all.nohup 2>&1 &
set -euo pipefail
source "$(dirname "$0")/_common.sh"
MODE=quality
if [ "${1:-}" = "--mode" ]; then MODE="$2"; shift 2; fi

python - <<'PY'
from ksent.config import frozen_path, global_cfg
missing = [f"{d}/{m}" for d in global_cfg()["datasets"] for m in global_cfg()["methods"]
           if not frozen_path(d, m).exists()]
if missing:
    raise SystemExit(f"missing frozen configs (run 04_tune.sh): {missing}")
PY
python -m ksent.scheduler --kind final --run-mode "$MODE" --stage "final_$MODE" "$@"
if [ "$MODE" = quality ]; then python -m ksent.analysis.check_results; fi
