# Memory Optimizations for Joint Video-Action Training

## Problem
CUDA OOM with 100352-dimensional text embeddings and 11.5B parameter VideoDiT model.

## Root Cause Analysis
1. **Text Embedding Dimension**: 100352 is the raw T5 embedding dimension
2. **Direct Cross-Attention**: Using 100352 dim directly in cross-attention causes huge memory usage
3. **No Mixed Precision**: FP32 training uses ~4x more memory than BF16
4. **Activation Storage**: 28 transformer layers store all activations for backward

## Implemented Optimizations

### 1. Cross-Attention Projection (CRITICAL)
**File**: `src/cosmost/models/dit.py`

Original code uses two-stage projection:
```python
# 100352 -> 1024 -> model_channels
use_crossattn_projection=True,
crossattn_proj_in_channels=100352,
crossattn_emb_channels=1024,
```

This reduces cross-attention memory from ~400MB per sequence to ~4MB.

### 2. Gradient Checkpointing
**File**: `src/cosmost/models/dit.py`

Enables `torch.utils.checkpoint` to trade compute for memory:
```python
use_gradient_checkpointing=True  # Saves ~50% activation memory
```

### 3. Automatic Mixed Precision (AMP)
**File**: `train.py`

Uses BF16/FP16 for forward/backward, FP32 for optimizer:
```python
use_amp=True  # Saves ~40-50% memory
```

### 4. Reduced Batch Size
**File**: `configs/default.py`

Start with batch_size=1 and gradually increase.

## Memory Estimates

| Component | FP32 | BF16 + GC | Savings |
|-----------|------|-----------|---------|
| Model Params | ~46 GB | ~23 GB | 50% |
| Activations (28 layers) | ~30 GB | ~15 GB | 50% |
| Optimizer States | ~92 GB | ~92 GB | 0% |
| **Total per GPU** | **~168 GB** | **~130 GB** | **~23%** |

With FSDP (parameter sharding across 4 GPUs):
- Model Params: ~6 GB per GPU
- Activations: ~15 GB per GPU
- Optimizer States: ~23 GB per GPU
- **Total per GPU**: ~44 GB (should fit in 79 GB)

## Additional Optimizations (if still OOM)

### 1. Use FSDP (Fully Sharded Data Parallel)
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
model = FSDP(model, ...)
```

### 2. Reduce Model Size (Temporary)
```python
# In configs/default.py
"video_dit": {
    "num_blocks": 14,  # Reduce from 28
    "model_channels": 1024,  # Reduce from 2048
}
```

### 3. Use 8-bit Optimizers
```python
import bitsandbytes as bnb
optimizer = bnb.optim.AdamW8bit(model.parameters(), ...)
```

### 4. Activation Checkpointing for Action Head
Add similar checkpointing to ActionHead if needed.

## Configuration Summary

Current optimized config:
```python
video_dit = {
    "crossattn_dim": 100352,       # Input dim
    "crossattn_emb_channels": 1024,  # Projected dim
    "use_crossattn_projection": True,
    "use_gradient_checkpointing": True,
}

training = {
    "batch_size": 1,
    "use_amp": True,
}
```

## Next Steps

1. Test with current optimizations
2. If still OOM, implement FSDP
3. Monitor actual memory usage with `nvidia-smi` or PyTorch profiler
