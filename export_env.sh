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
export WANDB_MODE="${WANDB_MODE:-online）}"
export JOB_WANDB_MODE="${JOB_WANDB_MODE:-$WANDB_MODE}"

# ── Action Head 训练配置 ─────────────────────────────────────────────────────
export ACTION_HEAD_ENABLED="${ACTION_HEAD_ENABLED:-1}"
export ACTION_HEAD_LR="${ACTION_HEAD_LR:-1e-4}"
export ACTION_HEAD_LOSS_WEIGHT="${ACTION_HEAD_LOSS_WEIGHT:-1.0}"
export ACTION_DELTA_VIDEO_T="${ACTION_DELTA_VIDEO_T:-0.5}"
export ACTION_HEAD_TIMESTEP_MODE="${ACTION_HEAD_TIMESTEP_MODE:-fixed}"  # beta / random / fixed
export ACTION_HEAD_FIXED_TIMESTEP="${ACTION_HEAD_FIXED_TIMESTEP:-900}"
export ACTION_HEAD_MIP_GT_MIX="${ACTION_HEAD_MIP_GT_MIX:-0.9}"
export ACTION_HEAD_NOISE_BETA_ALPHA="${ACTION_HEAD_NOISE_BETA_ALPHA:-1.5}"
export ACTION_HEAD_NOISE_BETA_BETA="${ACTION_HEAD_NOISE_BETA_BETA:-1.0}"
export ACTION_HEAD_NOISE_BETA_S="${ACTION_HEAD_NOISE_BETA_S:-0.999}"
# delta_v 空间 token 下采样（0 表示不池化，完整保留 HxW）
export ACTION_HEAD_DELTA_POOL_H="${ACTION_HEAD_DELTA_POOL_H:-0}"
export ACTION_HEAD_DELTA_POOL_W="${ACTION_HEAD_DELTA_POOL_W:-0}"
export ACTION_HEAD_DELTA_H="${ACTION_HEAD_DELTA_H:-60}"
export ACTION_HEAD_DELTA_W="${ACTION_HEAD_DELTA_W:-80}"
export ACTION_HEAD_ACTIONS_PER_LATENT="${ACTION_HEAD_ACTIONS_PER_LATENT:-8}"
export ACTION_HEAD_SAVE_EVERY="${ACTION_HEAD_SAVE_EVERY:-500}"
export ACTION_HEAD_SAVE_DIR="${ACTION_HEAD_SAVE_DIR:-/home/jwhe/linyihan/robot_posttrain/action_head_ckpt}"
# 可选：加载已有 action head 参数（仅 head，不影响 video ckpt）
export ACTION_HEAD_LOAD_PATH="${ACTION_HEAD_LOAD_PATH:-}"
export ACTION_HEAD_LOG_EVERY="${ACTION_HEAD_LOG_EVERY:-1}"
export ACTION_HEAD_WANDB_LOG="${ACTION_HEAD_WANDB_LOG:-1}"
export ACTION_STATE_USE_QNORM="${ACTION_STATE_USE_QNORM:-1}"
export ACTION_STATE_NORM_CLIP="${ACTION_STATE_NORM_CLIP:-1.0}"
# 只读一个全局 stats.json（包含所有 task 的 action/state q01/q99）
export ACTION_STATE_GLOBAL_STATS_JSON="${ACTION_STATE_GLOBAL_STATS_JSON:-/home/jwhe/linyihan/datasets/lerobot_robotwin_eef_clean_50/stats.json}"

# ── Video 初始化权重（DCP 目录）──────────────────────────────────────────────
export CHECKPOINT_LOAD_PATH="${CHECKPOINT_LOAD_PATH:-/home/jwhe/linyihan/robot_posttrain/open_loop/cosmos_diffusion_v2/robot_posttrain/my_video_experiment_20260323_133123/checkpoints/iter_000010000}"
export CHECKPOINT_LOAD_TRAINING_STATE="${CHECKPOINT_LOAD_TRAINING_STATE:-False}"
export CHECKPOINT_STRICT_RESUME="${CHECKPOINT_STRICT_RESUME:-True}"
export GRAD_ACCUM_ITER="${GRAD_ACCUM_ITER:-8}"
