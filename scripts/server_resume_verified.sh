#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
exec >> source/results/server/workflow.log 2>&1
trap 'result=$?; if [ "$result" -ne 0 ]; then conda run --no-capture-output -n ML python scripts/server_failure.py "$result"; fi' EXIT
export CUDA_HOME=/home/dung/miniconda3/envs/ML
export PATH="${CUDA_HOME}/bin:${PATH}"
export MAX_JOBS=2
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
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
printf 'Building causal-conv1d 1.6.0 with the API required by Mamba at %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
conda run --no-capture-output -n ML python -m pip install --no-cache-dir --no-deps --no-build-isolation --no-binary=causal-conv1d --force-reinstall causal-conv1d==1.6.0 > source/results/server/causal_source_build.log 2>&1
conda run --no-capture-output -n ML python -m pip check
./run_all.sh preflight --cloud-neural
# Let the separately started downloader finish before touching its manifests.
while conda run --no-capture-output -n ML python -c 'import psutil,sys; active=[]
for p in psutil.process_iter(["cmdline"]):
 try:
  a=p.info["cmdline"] or []
  if "korean_sentiment.cli" in a and "download" in a: active.append(p.pid)
 except (psutil.NoSuchProcess,psutil.AccessDenied): pass
sys.exit(0 if active else 1)'; do sleep 10; done
conda run --no-capture-output -n ML python scripts/server_execution.py prepare
conda run --no-capture-output -n ML python scripts/server_execution.py smoke
conda run --no-capture-output -n ML python scripts/server_execution.py pilot
conda run --no-capture-output -n ML python scripts/server_execution.py run
