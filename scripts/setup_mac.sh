#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/_common.sh"
CONDA_EXE_PATH="$(find_conda)"

if ! "${CONDA_EXE_PATH}" env list | awk '{print $1}' | grep -qx ML; then
  "${CONDA_EXE_PATH}" create -n ML python=3.11 -y
fi
"${CONDA_EXE_PATH}" env update -n ML -f "${REPO_ROOT}/environment.yml"
"${CONDA_EXE_PATH}" run -n ML python -m pip install --upgrade pip
"${CONDA_EXE_PATH}" run -n ML python -m pip install -r "${REPO_ROOT}/requirements-base.txt" -r "${REPO_ROOT}/requirements-mac.txt"
"${CONDA_EXE_PATH}" run -n ML python -m pip install --no-deps -e "${REPO_ROOT}"
ML_PREFIX="$("${CONDA_EXE_PATH}" run -n ML python -c 'import sys; print(sys.prefix)')"
TORCH_OMP="$("${CONDA_EXE_PATH}" run -n ML python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib", "libomp.dylib"))')"
if [ -f "${TORCH_OMP}" ]; then
  if "${CONDA_EXE_PATH}" list -n ML | grep -q '^llvm-openmp'; then
    "${CONDA_EXE_PATH}" remove -n ML llvm-openmp -y
  fi
  ln -sfn "${TORCH_OMP}" "${ML_PREFIX}/lib/libomp.dylib"
fi
"${CONDA_EXE_PATH}" run -n ML python -c 'import platform,sys,torch; assert sys.version_info[:2] == (3,11); print(platform.platform()); print("torch", torch.__version__, "mps", torch.backends.mps.is_available())'
mkdir -p "${SOURCE_ROOT}/results/artifacts"
if ! "${CONDA_EXE_PATH}" env export -n ML --from-history > "${SOURCE_ROOT}/results/artifacts/conda_from_history_mac.yml"; then
  cp "${REPO_ROOT}/environment.yml" "${SOURCE_ROOT}/results/artifacts/conda_from_history_mac.yml"
  printf '%s\n' "Conda could not export pip-interoperability history; saved the repository's minimal ML environment specification instead." >&2
fi
"${CONDA_EXE_PATH}" list -n ML > "${SOURCE_ROOT}/results/artifacts/conda_packages_mac.txt"
"${CONDA_EXE_PATH}" run -n ML python -m pip freeze > "${SOURCE_ROOT}/results/artifacts/pip_freeze_mac.txt"
"${CONDA_EXE_PATH}" run --no-capture-output -n ML python -m korean_sentiment.cli preflight
