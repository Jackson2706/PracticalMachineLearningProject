#!/usr/bin/env bash
# Shared prologue sourced by every entry-point script (portable: bash 3.2+, Linux & macOS):
#   activates conda env `ML`, cd's to source/, exports device/thread settings
#   (config, overridable by KSENT_DEVICE / KSENT_THREADS) and tees all output to
#   logs/<script>_<timestamp>.log.
set -euo pipefail

SCRIPT_NAME="$(basename "${BASH_SOURCE[1]:-$0}" .sh)"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SOURCE_DIR"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ML

export PYTHONPATH="$SOURCE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # single GPU
export PYTORCH_ENABLE_MPS_FALLBACK=1                        # fallbacks are recorded per run
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
# scratch/caches off the small overlay disk when /dev/shm is available (see logs/RUN_JOURNAL.md)
if [ -d /dev/shm ] && [ -w /dev/shm ]; then
  mkdir -p /dev/shm/ksent/tmp /dev/shm/ksent/cache
  export TMPDIR="${TMPDIR:-/dev/shm/ksent/tmp}" TECTONIC_CACHE_DIR="${TECTONIC_CACHE_DIR:-/dev/shm/ksent/cache/tectonic}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-/dev/shm/ksent/cache/matplotlib}"
fi

# Thread count (config or KSENT_THREADS, capped at the core count) - computed in Python.
N_THREADS="$(python -c 'from ksent.hardware import configured_threads; print(configured_threads())' 2>/dev/null || echo 8)"
export KSENT_THREADS="$N_THREADS"
export OMP_NUM_THREADS="$N_THREADS" MKL_NUM_THREADS="$N_THREADS" OPENBLAS_NUM_THREADS="$N_THREADS"
export VECLIB_MAXIMUM_THREADS="$N_THREADS" NUMEXPR_NUM_THREADS="$N_THREADS"
KSENT_DEVICE="${KSENT_DEVICE:-$(python -c 'from ksent.hardware import configured_device; print(configured_device())' 2>/dev/null || echo auto)}"
export KSENT_DEVICE

mkdir -p logs
LOG_FILE="logs/${SCRIPT_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${SCRIPT_NAME} | env=ML | device=${KSENT_DEVICE} | threads=${N_THREADS} | log=${LOG_FILE}"
