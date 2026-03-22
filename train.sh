#!/bin/bash
# =============================================================================
# 训练启动脚本
# =============================================================================
set -e

# 加载环境变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/export_env.sh"

# ★ 添加 cosmos-predict2.5 到 PYTHONPATH（可通过 COSMOS_PREDICT2_ROOT 覆盖）
export COSMOS_PREDICT2_ROOT="${COSMOS_PREDICT2_ROOT:-/home/jwhe/linyihan/cosmos-predict2.5-main}"
export PYTHONPATH="${COSMOS_PREDICT2_ROOT}:$PYTHONPATH"
# ── 参数配置 ────────────────────────────────────────────────────────────────
CONFIG="${1:-configs/config.py}"
NPROC="${NPROC:-1}"                          # GPU 数量，单卡用 1
MASTER_PORT="${MASTER_PORT:-12341}"
MAX_ITERS="${MAX_ITERS:-1000}"              # 调试阶段设小，正式训练设大

EXP_NAME="my_video_experiment"              # 与 configs/experiments/my_action_experiment.py 中注册名一致

# ── 训练命令 ────────────────────────────────────────────────────────────────
torchrun \
    --nproc_per_node=${NPROC} \
    --master_port=${MASTER_PORT} \
    -m scripts.train \
    --config="${CONFIG}" \
    -- \
    experiment=${EXP_NAME} \
    trainer.max_iter=${MAX_ITERS} \
    trainer.logging_iter=50 \
    trainer.validation_iter=200 \
    checkpoint.save_iter=500 \
    job.wandb_mode=disabled \
    job.name=${EXP_NAME}_$(date +%Y%m%d_%H%M%S)
