#!/usr/bin/env bash
# Every method x dataset at n=500, 1 seed, neural max 2 epochs. Smoke timings are never reported,
# so the cost-path runs skip the idle-machine wait (--no-exclusive-check).
#   * CPU, cost path (latency, size, memory): all 27 combinations   -> results_smoke/cpu/
#   * each accelerator, cost path: the 15 neural ones               -> results_smoke/<device>/
#   * each accelerator, quality mode through the GPU lanes (k_gpu)  -> results_smoke/<device>_quality/
# Usage: bash scripts/02_smoke_test.sh
set -euo pipefail
source "$(dirname "$0")/_common.sh"

python scripts/count_params.py
SMOKE="--kind smoke --max-epochs 2 --allow-fallback --progress-every 60"
# shellcheck disable=SC2086
python -m ksent.scheduler $SMOKE --run-mode cost --no-exclusive-check --device cpu \
  --out-root results_smoke/cpu --stage smoke_cpu
NEURAL="emb_mlp emb_lstm emb_gru small_transformer small_mamba"
ACCELS="$(python -m ksent.hardware --accelerators)"
for acc in $ACCELS; do
  echo "=== smoke test on accelerator: $acc ==="
  # shellcheck disable=SC2086
  python -m ksent.scheduler $SMOKE --run-mode cost --no-exclusive-check --device "$acc" --method $NEURAL \
    --out-root "results_smoke/$acc" --stage "smoke_$acc"
  # shellcheck disable=SC2086
  python -m ksent.scheduler $SMOKE --run-mode quality --device "$acc" --method $NEURAL --k-gpu 4 \
    --out-root "results_smoke/${acc}_quality" --stage "smoke_${acc}_quality"
done
python - "$ACCELS" <<'PY'
import glob, json, sys
ok = True
checks = [("cpu", 27, True)] + [(d, 15, c) for a in sys.argv[1].split() for d, c in ((a, True), (f"{a}_quality", False))]
for dev, expected, cost in checks:
    files = [f for f in glob.glob(f"results_smoke/{dev}/raw/*/*/*/*.json") if not f.endswith(".failed.json")]
    recs = [json.load(open(f)) for f in files]
    good = [r for r in recs if r.get("status") == "ok" and "test" in r
            and (("latency" in r and r["cost"].get("serialized_mb")) if cost else not r["timing_reportable"])]
    print(f"smoke [{dev}]: {len(good)}/{expected} successful")
    ok &= len(good) == expected
raise SystemExit(0 if ok else 1)
PY
