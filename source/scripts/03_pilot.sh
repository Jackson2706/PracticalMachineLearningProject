#!/usr/bin/env bash
# Default config per method at n=2,000 on one dataset (validation only) -> results_pilot/
# Reports time/memory per method, the Mamba backend check and runtime estimates.
# Usage: bash scripts/03_pilot.sh [--pilot-dataset nsmc]
set -euo pipefail
source "$(dirname "$0")/_common.sh"

python -m ksent.sweep --mode pilot "$@"
if python -c "from ksent.hardware import mamba_ssm_available as a; raise SystemExit(0 if a() else 1)"; then
  echo "mamba-ssm available -> also piloting small_mamba with the torch_ref backend for comparison"
  python -m ksent.sweep --mode pilot --method small_mamba --mamba-backend torch_ref \
    --out-root results_pilot/mamba_torch_ref "$@"
fi
python -m ksent.analysis.pilot_report
echo "Next: set 'mamba_backend:' in configs/global.yaml to the backend you choose, then run 04_tune.sh"
