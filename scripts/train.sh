#!/bin/bash
# Training script for Joint Video-Action Generation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse arguments
CONFIG="${CONFIG:-configs/default.py}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
RESUME="${RESUME:-}"

# Distributed training settings
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Environment
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "=========================================="
echo "Training Configuration:"
echo "  Config: ${CONFIG}"
echo "  Output: ${OUTPUT_DIR}"
echo "  GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "  NPROC: ${NPROC}"
echo "=========================================="

# Training command
if [ "${NPROC}" -eq 1 ]; then
    # Single GPU
    python "${PROJECT_ROOT}/train.py" \
        --config "${CONFIG}" \
        --output_dir "${OUTPUT_DIR}" \
        ${RESUME:+--resume "${RESUME}"}
else
    # Multi-GPU
    torchrun \
        --nproc_per_node="${NPROC}" \
        --master_port="${MASTER_PORT}" \
        "${PROJECT_ROOT}/train.py" \
        --config "${CONFIG}" \
        --output_dir "${OUTPUT_DIR}" \
        ${RESUME:+--resume "${RESUME}"}
fi

echo "Training completed!"
