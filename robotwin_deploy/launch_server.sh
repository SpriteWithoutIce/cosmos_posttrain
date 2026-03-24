#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

VIDEO_CKPT="${VIDEO_CKPT:-/home/jwhe/linyihan/robot_posttrain/open_loop/cosmos_diffusion_v2/robot_posttrain/my_video_experiment_action_20260323_200923/checkpoints/iter_000002000}"
ACTION_HEAD_CKPT="${ACTION_HEAD_CKPT:-/home/jwhe/linyihan/robot_posttrain/action_head_ckpt/action_head_iter_0002000.pt}"
VAE_PATH="${VAE_PATH:-/home/jwhe/linyihan/cosmos/tokenizer.pth}"
TEXT_EMB_PT="${TEXT_EMB_PT:-}"
STATS_JSON="${STATS_JSON:-}"
TEXT_EMB_DIM="${TEXT_EMB_DIM:-100352}"

cd "$(dirname "$0")/.."

python -m robotwin_deploy.cosmos_robotwin_server \
  --host "${HOST}" \
  --port "${PORT}" \
  --video_ckpt "${VIDEO_CKPT}" \
  --action_head_ckpt "${ACTION_HEAD_CKPT}" \
  --vae_path "${VAE_PATH}" \
  --text_emb_pt "${TEXT_EMB_PT}" \
  --stats_json "${STATS_JSON}" \
  --text_emb_dim "${TEXT_EMB_DIM}"

