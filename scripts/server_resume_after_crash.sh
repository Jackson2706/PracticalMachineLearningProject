#!/usr/bin/env bash
set -euo pipefail
cd /home/dung/PracticalMachineLearningProject
export CUDA_HOME=/home/dung/miniconda3/envs/ML
export PATH="$CUDA_HOME/bin:$PATH"
export KS_PRECISION=bf16 KS_CPU_THREADS=4 KS_NUM_WORKERS=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export KS_MIN_FREE_GIB=71
conda run --no-capture-output -n ML python scripts/server_recover_crash.py > source/results/server/crash_reproduction.log 2>&1
./run_all.sh preflight > source/results/server/recovery_preflight.log 2>&1
./run_all.sh smoke --device cuda > source/results/server/recovery_smoke.log 2>&1
exec conda run --no-capture-output -n ML python scripts/server_execution.py run >> source/results/server/workflow.log 2>&1
