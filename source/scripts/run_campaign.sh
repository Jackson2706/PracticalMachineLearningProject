#!/usr/bin/env bash
# Chains Stages C (after the pilot) -> D -> E -> F -> G -> H with no idle gaps.
# Every stop condition from the brief ends the chain (exit 2) and is journaled; resumable
# (each scheduler skips finished jobs), so re-running this script continues where it stopped.
#   nohup bash scripts/run_campaign.sh > logs/run_campaign.nohup 2>&1 &
set -euo pipefail
source "$(dirname "$0")/_common.sh"
J=logs/RUN_JOURNAL.md
note() { echo "- [$(date -u '+%F %T') UTC] run_campaign: $*" >> "$J"; echo "[$(date -u '+%F %T')] $*"; }
stop() { note "STOP: $*"; exit 2; }

# ---- C: wait for the pilot (+ kgpu probe + report), then freeze the scheduler settings
while pgrep -f "03_pilot.s[h]" >/dev/null; do sleep 30; done
[ -f results_pilot/kgpu_choice.json ] || stop "pilot finished without kgpu_choice.json (see logs/03_pilot.nohup)"
K=$(python -c "import json; print(int(json.load(open('results_pilot/kgpu_choice.json'))['k_gpu']))")
CPUJ=$(python - "$K" <<'PY'
import sys
from ksent.config import global_cfg
from ksent.scheduler import physical_cores, visible_gpus
g = global_cfg(); sc = g["scheduler"]; k = int(sys.argv[1])
res = k * len(visible_gpus()) * int(g["neural_threads"]) + int(sc["reserved_system_cores"])
print(max(1, (physical_cores() - res) // int(g["n_threads"])))
PY
)
sed -i -E "s/^(  k_gpu:) null/\1 $K/; s/^(  cpu_jobs:) null/\1 $CPUJ/" configs/global.yaml
note "frozen scheduler settings: k_gpu=$K cpu_jobs=$CPUJ (n_threads=8, neural_threads=2)"
python -m ksent.analysis.pilot_report --k-gpu "$K" --cpu-jobs "$CPUJ" > logs/pilot_report_frozen.txt
ROUTE=$(python - <<'PY'
import pandas as pd
e = pd.read_csv("results_pilot/pilot_estimate.csv").set_index("stage").hours
t = e["D tuning (exclusive, cost mode)"]
route = "quality" if t > 8 else "cost"
total = e["TOTAL (tuning exclusive)"] + (e["D tuning (quality route, parallel)"] if route == "quality" else 0)
print(route, round(total, 2))
PY
)
read -r TROUTE TOTAL_H <<< "$ROUTE"
note "pilot estimate: total ${TOTAL_H} h, tuning route=${TROUTE} (details: logs/pilot_report_frozen.txt)"
python -c "import sys; sys.exit(0 if float('$TOTAL_H') <= 48 else 1)" || stop "estimated total ${TOTAL_H} h > 48 h"

# ---- D: tuning (270 fits)
bash scripts/04_tune.sh --mode "$TROUTE"
n=$(python -c "
from ksent.config import frozen_path, global_cfg; g = global_cfg()
print(sum(frozen_path(d, m).exists() for d in g['datasets'] for m in g['methods']))")
[ "$n" -eq 27 ] || stop "only $n/27 frozen configs"
git -C .. add source/configs/frozen
[ -z "$(git -C .. status --porcelain -- source/configs/frozen)" ] || git -C .. commit -q -m "Freeze tuned configs (Stage D, route=$TROUTE)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015GxTjszoB2bncahqLuoT2c" -- source/configs/frozen || stop "git commit of frozen configs failed"
note "Stage D done: 27 frozen configs committed ($(git -C .. rev-parse --short HEAD))"

fail_frac() {  # failed jobs of a stage / its total, from the scheduler summary
  python - "$1" <<'PY'
import re, sys
txt = open(f"logs/scheduler_{sys.argv[1]}.log").read()
m = re.findall(r"finished (\d+) runs in .*; failed=(\d+)", txt)   # end-of-stage summary line
f, t = (int(m[-1][1]), int(m[-1][0]) + int(m[-1][1])) if m else (0, 1)
print(f"{f / max(1, t):.4f}")
PY
}
chk() { python -c "import sys; sys.exit(0 if float('$(fail_frac "$1")') <= 0.02 else 1)" || stop "$1: more than 2% of runs failed"; }

# ---- E + F: quality-mode sweep and ablation (parallel lanes)
bash scripts/05_run_all.sh --mode quality || stop "05_run_all / check_results failed"
chk final_quality; note "Stage E done (main sweep, quality)"
bash scripts/06_run_ablation.sh --mode quality || stop "06_run_ablation failed"
chk ablation_quality; note "Stage F done (ablation, quality)"

# ---- exclusive block: tuning replay (quality route only), then the cost pass
if [ "$TROUTE" = quality ]; then
  bash scripts/04_tune.sh --mode cost
  note "tuning replay in cost mode done"
fi
bash scripts/06b_cost_pass.sh || stop "06b_cost_pass failed or determinism check > 0.5%"
note "Stage G done (cost pass + determinism)"

# ---- H
bash scripts/07_analyze.sh || stop "07_analyze failed"
note "Stage H done; Stage I (report text, slides, demo, PDFs) is next"
