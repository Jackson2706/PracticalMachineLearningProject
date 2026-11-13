#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
mkdir -p source/results/server
export KS_LAUNCHER_PID="$$"
export KS_LAUNCHER_STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exec >> source/results/server/workflow.log 2>&1
trap 'result=$?; if [ "$result" -ne 0 ]; then conda run --no-capture-output -n ML python scripts/server_failure.py "$result"; fi' EXIT
conda run --no-capture-output -n ML python -c 'import json,os,pathlib; p=pathlib.Path("source/results/server/active_run.json"); t=p.with_suffix(".tmp"); t.write_text(json.dumps({"launcher_pid":int(os.environ["KS_LAUNCHER_PID"]),"tmux_session":"korean-sentiment-all","command":"bash scripts/server_launcher.sh","start_time":os.environ["KS_LAUNCHER_STARTED"],"primary_log":"source/results/server/workflow.log","phase":"waiting_for_torch_install","production_started":False},indent=2)); t.replace(p)'
printf 'Waiting for the already running PyTorch installation. No concurrent environment mutation.\n'
while ! rg -q '^Successfully installed ' source/results/server/install_torch_cu121.log; do
  if rg -q '^ERROR:' source/results/server/install_torch_cu121.log; then
    printf 'PyTorch installation failed. Preserve its log and diagnose before resuming.\n'
    exit 1
  fi
  sleep 10
done
bash scripts/server_setup_ml.sh
