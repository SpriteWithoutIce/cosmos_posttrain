#!/bin/bash
# Single source of truth for training parameters (lowercase only).

# ---------- paths ----------
cosmos_predict2_root="${cosmos_predict2_root:-/home/jwhe/linyihan/cosmos-predict2.5}"
cosmos_pt_ckpt="${cosmos_pt_ckpt:-/home/jwhe/linyihan/cosmos/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt}"
cosmos_tokenizer="${cosmos_tokenizer:-/home/jwhe/linyihan/cosmos/tokenizer.pth}"

dataset_root="${dataset_root:-/home/jwhe/linyihan/datasets/lerobot_robotwin_eef_clean_50}"
lerobot_root="${lerobot_root:-$dataset_root}"
latent_root="${latent_root:-/home/jwhe/linyihan/datasets/lerobot_latents}"
action_state_global_stats_json="${action_state_global_stats_json:-/home/jwhe/linyihan/datasets/lerobot_robotwin_eef_clean_50/stats.json}"

imaginaire_output_root="${imaginaire_output_root:-/home/jwhe/linyihan/robot_posttrain/open_loop}"
action_head_save_dir="${action_head_save_dir:-/home/jwhe/linyihan/robot_posttrain/action_head_ckpt}"
checkpoint_load_path="${checkpoint_load_path:-/home/jwhe/linyihan/robot_posttrain/open_loop/cosmos_diffusion_v2/robot_posttrain/my_video_experiment_20260323_133123/checkpoints/iter_000010000}"

# ---------- runtime ----------
config="${config:-configs/config.py}"
experiment_name="${experiment_name:-my_video_experiment}"
cuda_visible_devices="${cuda_visible_devices:-0,1,2,3}"
nproc="${nproc:-4}"
master_port="${master_port:-12342}"
max_iters="${max_iters:-1000}"
grad_accum_iter="${grad_accum_iter:-2}"
logging_iter="${logging_iter:-50}"
validation_iter="${validation_iter:-1000}"
checkpoint_save_iter="${checkpoint_save_iter:-1000}"

wandb_mode="${wandb_mode:-disabled}"    # disabled / offline / online

checkpoint_load_training_state="${checkpoint_load_training_state:-False}"
checkpoint_strict_resume="${checkpoint_strict_resume:-True}"

# ---------- action head ----------
video_action_conditioner_type="${video_action_conditioner_type:-mlp}"
action_head_enabled="${action_head_enabled:-1}"
action_head_type="${action_head_type:-flow_matching}"
action_head_lr="${action_head_lr:-2e-4}"
action_head_loss_weight="${action_head_loss_weight:-1.0}"
action_head_timestep_mode="${action_head_timestep_mode:-beta}"   # uniform / beta / fixed
action_head_fixed_timestep="${action_head_fixed_timestep:-0.0}"
action_head_noise_beta_alpha="${action_head_noise_beta_alpha:-1.5}"
action_head_noise_beta_beta="${action_head_noise_beta_beta:-1.0}"
action_head_noise_beta_s="${action_head_noise_beta_s:-0.999}"
action_head_actions_per_latent="${action_head_actions_per_latent:-8}"
action_head_use_state_condition="${action_head_use_state_condition:-0}"
action_head_stop_gradient="${action_head_stop_gradient:-1}"
action_head_video_hidden_pred_only="${action_head_video_hidden_pred_only:-0}"
action_head_save_every="${action_head_save_every:-500}"
action_head_load_path="${action_head_load_path:-}"
action_head_log_every="${action_head_log_every:-1}"
action_head_wandb_log="${action_head_wandb_log:-1}"
action_state_use_qnorm="${action_state_use_qnorm:-1}"
action_state_norm_clip="${action_state_norm_clip:-1.0}"

export_train_env() {
  export CUDA_VISIBLE_DEVICES="$cuda_visible_devices"
  export NPROC="$nproc"

  export COSMOS_PREDICT2_ROOT="$cosmos_predict2_root"
  export COSMOS_PT_CKPT="$cosmos_pt_ckpt"
  export COSMOS_TOKENIZER="$cosmos_tokenizer"

  export DATASET_ROOT="$dataset_root"
  export LEROBOT_ROOT="$lerobot_root"
  export LATENT_ROOT="$latent_root"
  export ACTION_STATE_GLOBAL_STATS_JSON="$action_state_global_stats_json"
  export IMAGINAIRE_OUTPUT_ROOT="$imaginaire_output_root"

  export WANDB_MODE="$wandb_mode"

  export VIDEO_ACTION_CONDITIONER_TYPE="$video_action_conditioner_type"
  export ACTION_HEAD_ENABLED="$action_head_enabled"
  export ACTION_HEAD_TYPE="$action_head_type"
  export ACTION_HEAD_LR="$action_head_lr"
  export ACTION_HEAD_LOSS_WEIGHT="$action_head_loss_weight"
  export ACTION_HEAD_TIMESTEP_MODE="$action_head_timestep_mode"
  export ACTION_HEAD_FIXED_TIMESTEP="$action_head_fixed_timestep"
  export ACTION_HEAD_NOISE_BETA_ALPHA="$action_head_noise_beta_alpha"
  export ACTION_HEAD_NOISE_BETA_BETA="$action_head_noise_beta_beta"
  export ACTION_HEAD_NOISE_BETA_S="$action_head_noise_beta_s"
  export ACTION_HEAD_ACTIONS_PER_LATENT="$action_head_actions_per_latent"
  export ACTION_HEAD_USE_STATE_CONDITION="$action_head_use_state_condition"
  export ACTION_HEAD_STOP_GRADIENT="$action_head_stop_gradient"
  export ACTION_HEAD_VIDEO_HIDDEN_PRED_ONLY="$action_head_video_hidden_pred_only"
  export ACTION_HEAD_SAVE_EVERY="$action_head_save_every"
  export ACTION_HEAD_SAVE_DIR="$action_head_save_dir"
  export ACTION_HEAD_LOAD_PATH="$action_head_load_path"
  export ACTION_HEAD_LOG_EVERY="$action_head_log_every"
  export ACTION_HEAD_WANDB_LOG="$action_head_wandb_log"
  export ACTION_STATE_USE_QNORM="$action_state_use_qnorm"
  export ACTION_STATE_NORM_CLIP="$action_state_norm_clip"

  export PYTHONPATH="$COSMOS_PREDICT2_ROOT:${PYTHONPATH:-}"
}
