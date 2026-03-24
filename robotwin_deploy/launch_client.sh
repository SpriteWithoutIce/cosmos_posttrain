#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/path/to/your/robotwin}"
TASK_NAME="${TASK_NAME:-adjust_bottle}"
TEST_NUM="${TEST_NUM:-100}"
SEED="${SEED:-0}"
SAVE_ROOT="${SAVE_ROOT:-./results/cosmos_robotwin}"
EXTRA_CONFIG="${EXTRA_CONFIG:-}"

cd "$(dirname "$0")/.."

CMD=(
  python -m robotwin_deploy.eval_polict_client_openpi
  --robowin_root "${ROBOTWIN_ROOT}"
  --task_name "${TASK_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --test_num "${TEST_NUM}"
  --seed "${SEED}"
  --save_root "${SAVE_ROOT}"
)

if [[ -n "${EXTRA_CONFIG}" ]]; then
  CMD+=(--extra_config "${EXTRA_CONFIG}")
fi

"${CMD[@]}"
