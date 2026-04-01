#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

VIDEO_CKPT="${VIDEO_CKPT:-/home/jwhe/linyihan/robot_posttrain/open_loop/cosmos_diffusion_v2/robot_posttrain/my_video_experiment_action_20260323_200923/checkpoints/iter_000002000}"
LOAD_EMA_TO_REG="${LOAD_EMA_TO_REG:-auto}"
ACTION_HEAD_CKPT="${ACTION_HEAD_CKPT:-/home/jwhe/linyihan/robot_posttrain/action_head_ckpt/action_head_iter_0002000.pt}"
VAE_PATH="${VAE_PATH:-/home/jwhe/linyihan/cosmos/tokenizer.pth}"
VAE_DEVICE="${VAE_DEVICE:-cuda}"
TEXT_EMB_PT="${TEXT_EMB_PT:-}"
STATS_JSON="${STATS_JSON:-}"
TEXT_EMB_DIM="${TEXT_EMB_DIM:-100352}"
OBS_H="${OBS_H:-480}"
OBS_W="${OBS_W:-640}"
NUM_STEPS="${NUM_STEPS:-1}"
VIDEO_T="${VIDEO_T:-0.5}"
ACTION_DENOISE_STEPS="${ACTION_DENOISE_STEPS:-2}"
ACTION_BLEND="${ACTION_BLEND:-0.9}"
ACTION_NOISE_SCALE="${ACTION_NOISE_SCALE:-0.1}"
ACTION_TIMESTEP="${ACTION_TIMESTEP:-0}"

cd "$(dirname "$0")/.."

EXTRA_ARGS=()
if [[ "${LOAD_EMA_TO_REG}" == "1" || "${LOAD_EMA_TO_REG}" == "true" || "${LOAD_EMA_TO_REG}" == "True" ]]; then
  EXTRA_ARGS+=(--load_ema_to_reg)
fi
if [[ "${LOAD_EMA_TO_REG}" == "0" || "${LOAD_EMA_TO_REG}" == "false" || "${LOAD_EMA_TO_REG}" == "False" ]]; then
  EXTRA_ARGS+=(--no_load_ema_to_reg)
fi

python -m robotwin_deploy.cosmos_robotwin_server \
  --host "${HOST}" \
  --port "${PORT}" \
  --video_ckpt "${VIDEO_CKPT}" \
  --action_head_ckpt "${ACTION_HEAD_CKPT}" \
  --vae_path "${VAE_PATH}" \
  --vae_device "${VAE_DEVICE}" \
  --text_emb_pt "${TEXT_EMB_PT}" \
  --stats_json "${STATS_JSON}" \
  --text_emb_dim "${TEXT_EMB_DIM}" \
  --obs_h "${OBS_H}" \
  --obs_w "${OBS_W}" \
  --num_steps "${NUM_STEPS}" \
  --video_t "${VIDEO_T}" \
  --use_video_v_at_t \
  --action_denoise_steps "${ACTION_DENOISE_STEPS}" \
  --action_blend "${ACTION_BLEND}" \
  --action_noise_scale "${ACTION_NOISE_SCALE}" \
  --action_timestep "${ACTION_TIMESTEP}" \
  "${EXTRA_ARGS[@]}"
