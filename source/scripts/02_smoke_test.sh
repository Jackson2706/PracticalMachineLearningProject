#!/usr/bin/env bash
# Every method x dataset at n=500, 1 seed, neural max 2 epochs.
#   * on CPU (--device cpu): all 27 combinations       -> results_smoke/cpu/
#   * on each available accelerator: the 15 neural ones -> results_smoke/<device>/
# Usage: bash scripts/02_smoke_test.sh [--parallel K]
set -euo pipefail
source "$(dirname "$0")/_common.sh"

python scripts/count_params.py
python -m ksent.sweep --mode smoke --device cpu --max-epochs 2 "$@"
NEURAL="emb_mlp emb_lstm emb_gru small_transformer small_mamba"
ACCELS="$(python -m ksent.hardware --accelerators)"
for acc in $ACCELS; do
  echo "=== smoke test on accelerator: $acc ==="
  # shellcheck disable=SC2086
  python -m ksent.sweep --mode smoke --device "$acc" --method $NEURAL --max-epochs 2 "$@"
done
python - "$ACCELS" <<'PY'
import glob, json, sys
ok = True
for dev, expected in [("cpu", 27)] + [(a, 15) for a in sys.argv[1].split()]:
    files = glob.glob(f"results_smoke/{dev}/raw/*/*/*/*.json")
    n = sum(1 for f in files if not f.endswith(".failed.json") and json.load(open(f)).get("status") == "ok")
    print(f"smoke [{dev}]: {n}/{expected} successful")
    ok &= n == expected
raise SystemExit(0 if ok else 1)
PY
