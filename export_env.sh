#!/bin/bash
# =============================================================================
# 环境变量配置 — 请根据你的实际路径修改
# =============================================================================

# ── Cosmos Predict2.5 post-train checkpoint ──────────────────────────────────
export COSMOS_PT_CKPT="${COSMOS_PT_CKPT:-/home/jwhe/linyihan/cosmos/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt}"

# ── VAE Tokenizer ───────────────────────────────────────────────────────────
export COSMOS_TOKENIZER="${COSMOS_TOKENIZER:-/home/jwhe/linyihan/cosmos/tokenizer.pth}"

# # mean_std.pt：与 tokenizer.pth 同目录，或从 HuggingFace 缓存获取
# # 如果不存在，请设置正确的路径，或在首次运行时让训练脚本自动下载
# _COSMOS_HF_CACHE="${HOME}/.cache/huggingface/hub/models--nvidia--Cosmos-Predict2.5-2B/snapshots"
# if [ -f "${_COSMOS_HF_CACHE}/a64a214a5ff6937c35ac32a41a7922442ccdf774/mean_std.pt" ]; then
#     export COSMOS_MEAN_STD="${_COSMOS_HF_CACHE}/a64a214a5ff6937c35ac32a41a7922442ccdf774/mean_std.pt"
# else
#     export COSMOS_MEAN_STD="${COSMOS_MEAN_STD:-}"   # 可留空，训练脚本会尝试自动解析
# fi

# ── 数据集 ──────────────────────────────────────────────────────────────────
export DATASET_ROOT="${DATASET_ROOT:-/home/jwhe/linyihan/datasets/lerobot_robotwin_eef_clean_50}"
export LEROBOT_ROOT="${LEROBOT_ROOT:-${DATASET_ROOT}}"
export LATENT_ROOT="${LATENT_ROOT:-/home/jwhe/linyihan/datasets/lerobot_latents}"

# ── 训练输出根目录 ────────────────────────────────────────────────────────────
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-/home/jwhe/linyihan/robot_posttrain/open_loop}"

# ── HuggingFace Token（如需下载受限 ckpt）────────────────────────────────────
# export HF_TOKEN="your_hf_token_here"

# ── 强制使用本地 HuggingFace 缓存（避免重复下载）─────────────────────────────
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_SYMLINKS=1

# ── 指向本地已下载的 Cosmos-Reason1-7B 模型 ─────────────────────────────────
export COSMOS_REASON1_DIR="/home/jwhe/linyihan/Cosmos-Reason1-7B"

# ── 其他推荐变量 ─────────────────────────────────────────────────────────────
export TOKENIZERS_PARALLELISM=false
export COSMOS_INTERNAL=0
export CUDA_MODULE_LOADING=LAZY

# wandb 模式（disabled / offline / online）
export WANDB_MODE="${WANDB_MODE:-disabled}"
export JOB_WANDB_MODE="${JOB_WANDB_MODE:-$WANDB_MODE}"

# ── Action Head 训练配置 ─────────────────────────────────────────────────────
export ACTION_HEAD_ENABLED="${ACTION_HEAD_ENABLED:-1}"
export ACTION_HEAD_LOSS_WEIGHT="${ACTION_HEAD_LOSS_WEIGHT:-1.0}"
export ACTION_HEAD_SAVE_EVERY="${ACTION_HEAD_SAVE_EVERY:-500}"
export ACTION_HEAD_SAVE_DIR="${ACTION_HEAD_SAVE_DIR:-/home/jwhe/linyihan/robot_posttrain/action_head_ckpt}"
# 可选：加载已有 action head 参数（仅 head，不影响 video ckpt）
export ACTION_HEAD_LOAD_PATH="${ACTION_HEAD_LOAD_PATH:-}"
export ACTION_HEAD_LOG_EVERY="${ACTION_HEAD_LOG_EVERY:-50}"
export ACTION_HEAD_WANDB_LOG="${ACTION_HEAD_WANDB_LOG:-1}"
