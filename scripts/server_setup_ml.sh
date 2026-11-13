#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export CUDA_HOME=/home/dung/miniconda3/envs/ML
export PATH="${CUDA_HOME}/bin:${PATH}"
export PIP_CONSTRAINT="${ROOT}/source/results/server/server_constraints.txt"
export PIP_NO_CACHE_DIR=1
export MAX_JOBS=2
export KS_CPU_THREADS=4
export KS_NUM_WORKERS=0
export KS_MAX_PARALLEL_CPU_JOBS=1
export KS_MAX_PARALLEL_GPU_JOBS=1
export KS_PRECISION=bf16
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Dependencies were installed and CUDA extensions rebuilt for this host's glibc.
# Verify them on resume instead of reinstalling the incompatible upstream wheel.
./run_all.sh preflight --cloud-neural
conda run --no-capture-output -n ML python -m pip check
conda run --no-capture-output -n ML python -m korean_sentiment.cli --help
conda run --no-capture-output -n ML python -m korean_sentiment.cli pilot --help
conda run --no-capture-output -n ML python -c 'import sys, torch, xgboost, sklearn; from korean_sentiment.orchestration.preflight import mamba_smoke; print(sys.executable, sys.version); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); x=torch.ones(16,device="cuda"); print("CUDA tensor sum", x.sum().item()); print("Mamba", mamba_smoke())'
conda run --no-capture-output -n ML python scripts/server_execution.py prepare
conda run --no-capture-output -n ML python scripts/server_execution.py smoke
conda run --no-capture-output -n ML python scripts/server_execution.py pilot
conda run --no-capture-output -n ML python scripts/server_execution.py run
