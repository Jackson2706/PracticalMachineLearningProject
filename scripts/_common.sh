#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -d "${REPO_ROOT}/Source/src/korean_sentiment" ]; then
  SOURCE_ROOT="${REPO_ROOT}/Source"
else
  SOURCE_ROOT="${REPO_ROOT}/source"
fi

find_conda() {
  if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then
    printf '%s\n' "${CONDA_EXE}"
  elif command -v conda >/dev/null 2>&1; then
    command -v conda
  else
    printf '%s\n' "Conda or Miniconda is required. Install it, then run: conda create -n ML python=3.11 -y" >&2
    return 1
  fi
}

run_ml() {
  local conda_exe
  conda_exe="$(find_conda)"
  if ! "${conda_exe}" env list | awk '{print $1}' | grep -qx ML; then
    printf '%s\n' "The ML environment does not exist. Run: conda create -n ML python=3.11 -y" >&2
    return 2
  fi
  "${conda_exe}" run --no-capture-output -n ML "$@"
}

run_cli() {
  run_ml python -m korean_sentiment.cli "$@"
}

timestamped_log() {
  local name="$1"
  mkdir -p "${SOURCE_ROOT}/logs"
  printf '%s/%s_%s.log\n' "${SOURCE_ROOT}/logs" "${name}" "$(date -u +%Y%m%dT%H%M%SZ)"
}
