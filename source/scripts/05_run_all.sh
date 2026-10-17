#!/usr/bin/env bash
# 675 main fits (3 x 9 x 5 sizes x 5 seeds, kiwi); dataset -> method -> size -> seed; resumable.
# Run on an otherwise idle machine - timings are only reportable for sequential runs.
# Usage: bash scripts/05_run_all.sh [--dataset nsmc ...] [--method tfidf_lr ...] [--seeds 13 42 ...]
# Long run:  nohup bash scripts/05_run_all.sh > logs/run_all.nohup 2>&1 &
set -euo pipefail
source "$(dirname "$0")/_common.sh"

python - <<'PY'
from ksent.config import frozen_path, global_cfg
missing = [f"{d}/{m}" for d in global_cfg()["datasets"] for m in global_cfg()["methods"]
           if not frozen_path(d, m).exists()]
if missing:
    raise SystemExit(f"missing frozen configs (run 04_tune.sh): {missing}")
PY
python -m ksent.sweep --mode final "$@"
python -m ksent.analysis.check_results
