#!/bin/bash
# =============================================================================
# 训练启动脚本
# =============================================================================
set -e
export CUDA_VISIBLE_DEVICES=2,3
export WANDB_MODE=online
export JOB_WANDB_MODE=online
# 加载环境变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/export_env.sh"
mkdir -p "${IMAGINAIRE_OUTPUT_ROOT}"

# ★ 添加 cosmos-predict2.5 到 PYTHONPATH（可通过 COSMOS_PREDICT2_ROOT 覆盖）
export COSMOS_PREDICT2_ROOT="${COSMOS_PREDICT2_ROOT:-/home/jwhe/linyihan/cosmos-predict2.5}"
export PYTHONPATH="${COSMOS_PREDICT2_ROOT}:$PYTHONPATH"
# ── 参数配置 ────────────────────────────────────────────────────────────────
CONFIG="${1:-configs/config.py}"
NPROC="${NPROC:-2}"                          # GPU 数量，单卡用 1
MASTER_PORT="${MASTER_PORT:-12342}"
MAX_ITERS="${MAX_ITERS:-40000}"              # 调试阶段设小，正式训练设大
JOB_WANDB_MODE="${JOB_WANDB_MODE:-disabled}"
GRAD_ACCUM_ITER="${GRAD_ACCUM_ITER:-1}"
CHECKPOINT_LOAD_PATH="${CHECKPOINT_LOAD_PATH:-/home/jwhe/linyihan/robot_posttrain/open_loop/cosmos_diffusion_v2/robot_posttrain/my_video_experiment_20260323_133123/checkpoints/iter_000010000}"
CHECKPOINT_LOAD_TRAINING_STATE="${CHECKPOINT_LOAD_TRAINING_STATE:-False}"
CHECKPOINT_STRICT_RESUME="${CHECKPOINT_STRICT_RESUME:-True}"

EXP_NAME="my_video_experiment"              # 与 configs/experiments/my_action_experiment.py 中注册名一致

# ── 训练命令 ────────────────────────────────────────────────────────────────
torchrun \
    --nproc_per_node=${NPROC} \
    --master_port=${MASTER_PORT} \
    -m scripts.train \
    --config="${CONFIG}" \
    -- \
    experiment=${EXP_NAME} \
    checkpoint.load_path=${CHECKPOINT_LOAD_PATH} \
    checkpoint.load_training_state=${CHECKPOINT_LOAD_TRAINING_STATE} \
    checkpoint.strict_resume=${CHECKPOINT_STRICT_RESUME} \
    trainer.max_iter=${MAX_ITERS} \
    trainer.grad_accum_iter=${GRAD_ACCUM_ITER} \
    trainer.logging_iter=50 \
    trainer.validation_iter=5000 \
    checkpoint.save_iter=10000 \
    job.wandb_mode=${JOB_WANDB_MODE} \
    job.name=${EXP_NAME}_action_$(date +%Y%m%d_%H%M%S) \
    # > logs/train_${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log 2>&1
