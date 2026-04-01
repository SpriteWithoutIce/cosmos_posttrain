#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/train_params.sh"
export_train_env

mkdir -p "${IMAGINAIRE_OUTPUT_ROOT}" logs

run_name="${experiment_name}_action_$(date +%Y%m%d_%H%M%S)"
log_file="logs/train_${run_name}.log"

torchrun \
  --nproc_per_node="${nproc}" \
  --master_port="${master_port}" \
  -m scripts.train \
  --config="${config}" \
  -- \
  experiment="${experiment_name}" \
  checkpoint.load_path="${checkpoint_load_path}" \
  checkpoint.load_training_state="${checkpoint_load_training_state}" \
  checkpoint.strict_resume="${checkpoint_strict_resume}" \
  trainer.max_iter="${max_iters}" \
  trainer.grad_accum_iter="${grad_accum_iter}" \
  trainer.logging_iter="${logging_iter}" \
  trainer.validation_iter="${validation_iter}" \
  checkpoint.save_iter="${checkpoint_save_iter}" \
  job.wandb_mode="${wandb_mode}" \
  job.name="${run_name}" \
  # > "${log_file}" 2>&1

echo "Training started. Log: ${log_file}"
