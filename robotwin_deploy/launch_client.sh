#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/path/to/your/robotwin}"
TASK_NAME="${TASK_NAME:-adjust_bottle}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
CLIENT_CONFIG="${CLIENT_CONFIG:-/root/linyihan/cosmos_posttrain/policy/ACT/deploy_policy.yml}"
TEST_NUM="${TEST_NUM:-100}"
SEED="${SEED:-0}"
SAVE_ROOT="${SAVE_ROOT:-./results/cosmos_robotwin}"

cd "$(dirname "$0")/.."

CMD=(
  python -m robotwin_deploy.eval_polict_client_openpi
  --config "${CLIENT_CONFIG}"
  --tasks "${TASK_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --test_num "${TEST_NUM}"
  --save_root "${SAVE_ROOT}"
  --overrides
  --task_config "${TASK_CONFIG}"
  --seed "${SEED}"
)

ROBOTWIN_ROOT="${ROBOTWIN_ROOT}" "${CMD[@]}"
