#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/_common.sh"
CONDA_EXE_PATH="$(find_conda)"

if [ "$(uname -s)" != "Linux" ]; then
  printf '%s\n' "setup-cloud requires Linux." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf '%s\n' "setup-cloud requires an NVIDIA driver and nvidia-smi." >&2
  exit 2
fi
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv
if ! "${CONDA_EXE_PATH}" env list | awk '{print $1}' | grep -qx ML; then
  "${CONDA_EXE_PATH}" create -n ML python=3.11 -y
fi
"${CONDA_EXE_PATH}" env update -n ML -f "${REPO_ROOT}/environment.yml"
"${CONDA_EXE_PATH}" run -n ML python -m pip install --upgrade pip
"${CONDA_EXE_PATH}" run -n ML python -m pip install -r "${REPO_ROOT}/requirements-base.txt"
if [ -n "${KS_TORCH_INDEX_URL:-}" ]; then
  "${CONDA_EXE_PATH}" run -n ML python -m pip install 'torch>=2.3,<3' --index-url "${KS_TORCH_INDEX_URL}"
else
  "${CONDA_EXE_PATH}" run -n ML python -m pip install 'torch>=2.3,<3'
fi
if [ -n "${KS_CAUSAL_CONV1D_WHEEL:-}" ]; then
  "${CONDA_EXE_PATH}" run -n ML python -m pip install --no-build-isolation "${KS_CAUSAL_CONV1D_WHEEL}"
fi
"${CONDA_EXE_PATH}" run -n ML python -m pip install --no-build-isolation -r "${REPO_ROOT}/requirements-cloud.txt"
MAMBA_KEEP_CUDA_BUILD=TRUE "${CONDA_EXE_PATH}" run -n ML python -m pip install --no-build-isolation "${KS_MAMBA_WHEEL:-mamba-ssm>=2.2,<3}"
"${CONDA_EXE_PATH}" run -n ML python -m pip install --no-deps -e "${REPO_ROOT}"
mkdir -p "${SOURCE_ROOT}/results/artifacts"
if ! "${CONDA_EXE_PATH}" env export -n ML --from-history > "${SOURCE_ROOT}/results/artifacts/conda_from_history_cloud.yml"; then
  cp "${REPO_ROOT}/environment.yml" "${SOURCE_ROOT}/results/artifacts/conda_from_history_cloud.yml"
  printf '%s\n' "Conda could not export pip-interoperability history; saved the repository's minimal ML environment specification instead." >&2
fi
"${CONDA_EXE_PATH}" list -n ML > "${SOURCE_ROOT}/results/artifacts/conda_packages_cloud.txt"
"${CONDA_EXE_PATH}" run -n ML python -m pip freeze > "${SOURCE_ROOT}/results/artifacts/pip_freeze_cloud.txt"
"${CONDA_EXE_PATH}" run --no-capture-output -n ML python -m korean_sentiment.cli preflight --cloud-neural
