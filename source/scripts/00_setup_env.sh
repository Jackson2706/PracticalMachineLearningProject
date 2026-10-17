#!/usr/bin/env bash
# Install pinned requirements into the existing conda env `ML` (never creates an env).
# PyTorch is installed by YOU for your hardware; this script only checks it and prints
# the right command if it is missing.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

if ! python -c "import torch" >/dev/null 2>&1; then
  echo "PyTorch is not installed in env ML. Install the build for your hardware, then re-run:"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  NVIDIA GPU detected (driver CUDA: $(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' || echo unknown)):"
    echo "    pip install torch --index-url https://download.pytorch.org/whl/cu124   # pick the cuXXX matching your driver"
  elif [ "$(uname)" = "Darwin" ]; then
    echo "  macOS / Apple Silicon (MPS):"
    echo "    pip install torch"
  else
    echo "  CPU only:"
    echo "    pip install torch --index-url https://download.pytorch.org/whl/cpu"
  fi
  exit 1
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [ "$(uname)" = "Darwin" ]; then
  # macOS: xgboost's wheel needs an OpenMP runtime (libomp.dylib) but does not ship one,
  # while torch bundles its own. Two different libomp copies in one process segfault
  # (exit 139) once both run multithreaded code, so point xgboost at torch's copy.
  TORCH_OMP="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib", "libomp.dylib"))')"
  if [ -f "$TORCH_OMP" ]; then
    if conda list -n ML | grep -q '^llvm-openmp'; then conda remove -y -n ML llvm-openmp; fi
    ln -sfn "$TORCH_OMP" "$CONDA_PREFIX/lib/libomp.dylib"
    echo "libomp: $CONDA_PREFIX/lib/libomp.dylib -> $TORCH_OMP"
  fi
fi

if [ "$(uname)" = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
  echo "Linux + NVIDIA GPU -> trying mamba-ssm (failure is tolerated; torch_ref backend is the fallback)"
  python -m pip install causal-conv1d mamba-ssm --no-build-isolation || echo "mamba-ssm install failed -> torch_ref only"
else
  echo "mamba-ssm skipped (needs Linux + CUDA); small_mamba will use the torch_ref backend"
fi

python - <<'PY'
import importlib
import ksent  # noqa: F401  (imports xgboost before torch)
for m in ["numpy", "pandas", "pyarrow", "sklearn", "xgboost", "torch", "kiwipiepy", "yaml", "psutil",
          "matplotlib", "joblib", "gradio", "pytest"]:
    mod = importlib.import_module(m)
    print(f"  ok  {m:12s} {getattr(mod, '__version__', '')}")
from kiwipiepy import Kiwi
print("  kiwi test:", [f"{t.form}/{t.tag}" for t in Kiwi(typos=None).tokenize("영화 안 봤는데 좋네요 ㅋㅋ")])
PY

python -m ksent.hardware --dump-env results/env | tail -1
echo "Environment ready."
