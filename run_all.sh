#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
. "${REPO_ROOT}/scripts/_common.sh"

usage() {
  printf '%s\n' "Usage: ./run_all.sh {setup-mac|setup-cloud|preflight|download|preprocess|test|smoke|pilot|plan|tune|full|aggregate|bonus-plan|bonus-tune|bonus-full|bonus-aggregate|benchmark-all|bonus-all|both|status|archive-results|all} [options]"
}

confirmed=0
forwarded=("_sentinel")
for argument in "${@:2}"; do
  case "${argument}" in
    --confirm-expensive|--noninteractive-confirm-expensive) confirmed=1 ;;
    *) forwarded+=("${argument}") ;;
  esac
done

require_confirmation() {
  if [ "${confirmed}" -ne 1 ]; then
    printf '%s\n' "This command starts paid or long-running experiments. Re-run with --confirm-expensive, or --noninteractive-confirm-expensive inside tmux/nohup." >&2
    exit 2
  fi
}

command_name="${1:-}"
case "${command_name}" in
  setup-mac) exec "${REPO_ROOT}/scripts/setup_mac.sh" "${forwarded[@]:1}" ;;
  setup-cloud) exec "${REPO_ROOT}/scripts/setup_cloud.sh" "${forwarded[@]:1}" ;;
  preflight) run_cli preflight "${forwarded[@]:1}" 2>&1 | tee "$(timestamped_log preflight)" ;;
  download) run_cli download "${forwarded[@]:1}" 2>&1 | tee "$(timestamped_log download)" ;;
  preprocess) run_cli preprocess "${forwarded[@]:1}" 2>&1 | tee "$(timestamped_log preprocess)" ;;
  test) run_ml python -m pytest \
    "${SOURCE_ROOT}/tests/test_ks_data.py" \
    "${SOURCE_ROOT}/tests/test_models.py" \
    "${SOURCE_ROOT}/tests/test_contrastive.py" \
    "${SOURCE_ROOT}/tests/test_orchestration.py" \
    "${forwarded[@]:1}" 2>&1 | tee "$(timestamped_log test)" ;;
  smoke) run_cli smoke "${forwarded[@]:1}" 2>&1 | tee "$(timestamped_log smoke)" ;;
  pilot) run_cli pilot "${forwarded[@]:1}" 2>&1 | tee "$(timestamped_log pilot)" ;;
  plan) run_cli plan "${forwarded[@]:1}" 2>&1 | tee "$(timestamped_log plan)" ;;
  tune) require_confirmation; run_cli tune "${forwarded[@]:1}" 2>&1 | tee "$(timestamped_log tune)" ;;
  full) require_confirmation; run_cli full "${forwarded[@]:1}" 2>&1 | tee "$(timestamped_log full)" ;;
  aggregate) run_cli aggregate "${forwarded[@]:1}" 2>&1 | tee "$(timestamped_log aggregate)" ;;
  bonus-plan) run_cli bonus-plan "${forwarded[@]:1}" 2>&1 | tee "$(timestamped_log bonus_plan)" ;;
  bonus-tune) require_confirmation; run_cli bonus-tune "${forwarded[@]:1}" 2>&1 | tee "$(timestamped_log bonus_tune)" ;;
  bonus-full) require_confirmation; run_cli bonus-full "${forwarded[@]:1}" 2>&1 | tee "$(timestamped_log bonus_full)" ;;
  bonus-aggregate) run_cli bonus-aggregate "${forwarded[@]:1}" 2>&1 | tee "$(timestamped_log bonus_aggregate)" ;;
  status) run_cli status "${forwarded[@]:1}" ;;
  archive-results) run_cli archive-results "${forwarded[@]:1}" 2>&1 | tee "$(timestamped_log archive_results)" ;;
  benchmark-all)
    require_confirmation
    run_cli preflight
    run_cli download
    run_cli preprocess
    "${REPO_ROOT}/run_all.sh" test
    run_cli smoke
    run_cli plan
    run_cli pilot --max-jobs 1
    run_cli tune "${forwarded[@]:1}"
    run_cli full "${forwarded[@]:1}"
    run_cli aggregate
    ;;
  bonus-all)
    require_confirmation
    run_cli bonus-plan
    run_cli bonus-tune "${forwarded[@]:1}"
    run_cli bonus-full "${forwarded[@]:1}"
    run_cli bonus-aggregate
    ;;
  both)
    require_confirmation
    "${REPO_ROOT}/run_all.sh" benchmark-all --noninteractive-confirm-expensive "${forwarded[@]:1}"
    "${REPO_ROOT}/run_all.sh" bonus-all --noninteractive-confirm-expensive "${forwarded[@]:1}"
    ;;
  all)
    require_confirmation
    "${REPO_ROOT}/run_all.sh" both --noninteractive-confirm-expensive "${forwarded[@]:1}"
    "${REPO_ROOT}/run_all.sh" archive-results
    ;;
  *) usage; exit 2 ;;
esac
