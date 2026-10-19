#!/usr/bin/env bash
# Stage G cost pass (exclusive, sequential) -> results/cost/raw/:
#   1. n=8,000 x 9 methods x 3 datasets x 5 seeds (Table 4, Fig 3)
#   2. crossover sizes < 8,000: best TF-IDF + best neural x 5 seeds (Table 5; needs crossover.csv)
#   3. determinism check: cost vs quality predictions
set -euo pipefail
source "$(dirname "$0")/_common.sh"

python -m ksent.scheduler --kind final --run-mode cost --sizes 8000 --stage cost_n8000 "$@"
python -m ksent.analysis.aggregate && python -m ksent.analysis.crossover
python -m ksent.scheduler --kind crossover_cost --run-mode cost --stage cost_crossover "$@"
python -m ksent.analysis.determinism
