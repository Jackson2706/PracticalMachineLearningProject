#!/usr/bin/env bash
# Pilot (cost mode, exclusive): default config per method at n=2,000 and n=8,000 on one dataset
# (validation only) -> results_pilot/. Then the K_gpu probe (neural jobs per GPU) and the report:
# per-method time/memory, Mamba backend check, scheduler estimates, wall-time estimate per stage.
# Usage: bash scripts/03_pilot.sh [--pilot-dataset nsmc]
set -euo pipefail
source "$(dirname "$0")/_common.sh"

python -m ksent.scheduler --kind pilot --run-mode cost --sizes 2000 8000 --out-root results_pilot --stage pilot "$@"
if python -c "from ksent.hardware import mamba_ssm_available as a; raise SystemExit(0 if a() else 1)"; then
  echo "mamba-ssm available -> also piloting small_mamba with the torch_ref backend for comparison"
  python -m ksent.scheduler --kind pilot --run-mode cost --sizes 2000 8000 --method small_mamba \
    --mamba-backend torch_ref --out-root results_pilot/mamba_torch_ref --stage pilot_mamba_torch_ref "$@"
fi
if [ -n "$(python -m ksent.hardware --accelerators)" ]; then
  python -m ksent.analysis.kgpu_probe
fi
python -m ksent.analysis.pilot_report
echo "Next: freeze k_gpu / cpu_jobs / mamba_backend in configs/global.yaml, then run 04_tune.sh"
