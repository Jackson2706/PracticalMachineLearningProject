#!/usr/bin/env bash
# 270 tuning fits (3 datasets x 9 methods x 10 configs, n=2,000, seed 0, kiwi) -> configs/frozen/
# Tuning time is reported, so tuning runs in cost mode (exclusive, sequential) by default.
#   --mode cost     (default) exclusive tuning -> results/tuning/ -> select + freeze
#   --mode quality  parallel tuning -> results/tuning_quality/ -> select + freeze from it; the
#                   270 fits must then be replayed with --mode cost to measure tuning time
#                   (the replay re-runs the same configs; selection stays the quality one and is
#                   checked to be identical)
# Usage: bash scripts/04_tune.sh [--mode cost|quality] [--dataset ...] [--method ...]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
MODE=cost
if [ "${1:-}" = "--mode" ]; then MODE="$2"; shift 2; fi

python -m ksent.scheduler --kind tuning --run-mode "$MODE" --stage "tuning_$MODE" "$@"
if [ "$MODE" = quality ]; then
  python -m ksent.tune --source tuning_quality
  echo "Selection frozen from quality tuning. Replay for tuning time: bash scripts/04_tune.sh --mode cost"
else
  if [ -d results/tuning_quality/tuning ]; then
    python -m ksent.tune --source tuning_quality --times-from tuning   # keep quality selection, add cost times
  else
    python -m ksent.tune
  fi
fi
echo "Frozen configs written to configs/frozen/. Commit them before the main sweep."
