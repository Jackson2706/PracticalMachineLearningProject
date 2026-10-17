#!/usr/bin/env bash
# Tokenizer ablation: best TF-IDF + best neural method per dataset (validation, n=8,000, main
# sweep) x {whitespace, kiwi, char} x {500, 8000} x 5 seeds. The 60 kiwi cells already exist,
# so 120 new fits. Requires the complete main sweep.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

python -m ksent.analysis.check_results
python -m ksent.sweep --mode ablation "$@"
python -m ksent.analysis.check_results --ablation
