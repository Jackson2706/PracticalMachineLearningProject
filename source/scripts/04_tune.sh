#!/usr/bin/env bash
# 270 tuning fits (3 datasets x 9 methods x 10 configs, n=2,000, seed 0, kiwi) -> configs/frozen/
# Usage: bash scripts/04_tune.sh [--parallel K] [--dataset ...] [--method ...]
set -euo pipefail
source "$(dirname "$0")/_common.sh"

python -m ksent.sweep --mode tuning "$@"
python -m ksent.tune
echo "Frozen configs written to configs/frozen/. Commit them before the main sweep:"
echo "  git add configs/frozen && git commit -m 'Freeze tuned configs'"
