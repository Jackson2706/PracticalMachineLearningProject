#!/usr/bin/env bash
# Tokenizer ablation (quality mode): best TF-IDF + best neural method per dataset (validation,
# n=8,000, main sweep) x {whitespace, kiwi, char} x {500, 8000} x 5 seeds. The 60 kiwi cells
# already exist, so 120 new fits. Requires the complete main sweep.
# Usage: bash scripts/06_run_ablation.sh [--mode quality]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
MODE=quality
if [ "${1:-}" = "--mode" ]; then MODE="$2"; shift 2; fi

python -m ksent.analysis.check_results
python -m ksent.scheduler --kind ablation --run-mode "$MODE" --stage "ablation_$MODE" "$@"
python -m ksent.analysis.check_results --ablation
