#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
exec >> source/results/server/workflow.log 2>&1
trap 'result=$?; if [ "$result" -ne 0 ]; then conda run --no-capture-output -n ML python scripts/server_failure.py "$result"; fi' EXIT
export CUDA_HOME=/home/dung/miniconda3/envs/ML
export PATH="${CUDA_HOME}/bin:${PATH}"
export MAX_JOBS=2
export MAMBA_FORCE_BUILD=TRUE
export MAMBA_KEEP_CUDA_BUILD=TRUE
export TORCH_CUDA_ARCH_LIST=8.9
export KS_PRECISION=bf16
export KS_CPU_THREADS=4
export KS_NUM_WORKERS=0
export KS_MAX_PARALLEL_CPU_JOBS=1
export KS_MAX_PARALLEL_GPU_JOBS=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
printf 'Rebuilding maintained Mamba 2.2.6.post3 for host glibc at %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvcc --version
conda run --no-capture-output -n ML python -m pip install --no-cache-dir --no-deps --no-build-isolation --no-binary=mamba-ssm --force-reinstall mamba-ssm==2.2.6.post3 > source/results/server/mamba_source_build.log 2>&1
conda run --no-capture-output -n ML python -m pip check
./run_all.sh preflight --cloud-neural
conda run --no-capture-output -n ML python scripts/server_execution.py prepare
conda run --no-capture-output -n ML python scripts/server_execution.py smoke
conda run --no-capture-output -n ML python scripts/server_execution.py pilot
conda run --no-capture-output -n ML python scripts/server_execution.py run
