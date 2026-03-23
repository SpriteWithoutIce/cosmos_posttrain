#!/bin/bash
# Backward-compatible wrapper.
# Real parameters are centralized in train_params.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/train_params.sh"
export_train_env
