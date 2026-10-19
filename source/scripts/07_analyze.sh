#!/usr/bin/env bash
# aggregate -> crossover -> ablation -> tables -> figures -> numbers.tex (works with partial results)
# Usage: bash scripts/07_analyze.sh [--placeholder] [--allow-mixed-hardware]
set -euo pipefail
source "$(dirname "$0")/_common.sh"

AGG_ARGS=""
FIG_ARGS=""
for a in "$@"; do
  case "$a" in
    --placeholder) FIG_ARGS="--placeholder" ;;
    --allow-mixed-hardware) AGG_ARGS="--allow-mixed-hardware" ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done
# shellcheck disable=SC2086
python -m ksent.analysis.aggregate $AGG_ARGS
python -m ksent.analysis.crossover
python -m ksent.analysis.ablation
python -m ksent.analysis.compute_usage
python -m ksent.analysis.make_tables
# shellcheck disable=SC2086
python -m ksent.analysis.make_figures $FIG_ARGS
python -m ksent.analysis.make_numbers
python -m ksent.analysis.make_summary
