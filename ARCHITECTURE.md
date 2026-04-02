# CosmosT Architecture

Self-contained Video-Action Joint Generation based on Cosmos-Predict2.5 + FastWAM-style design.

## Directory Structure

```
cosmos_posttrain/
├── src/cosmost/              # Main source code
│   ├── models/
│   │   ├── dit.py           # VideoDiT (simplified from minimal_v4_dit)
│   │   ├── action_head.py   # ActionHead (transformer-based)
│   │   └── joint_model.py   # JointVideoActionModel (main model)
│   ├── datasets/
│   │   └── lerobot_dataset.py  # LeRobotLatentDataset
│   ├── schedulers/
│   │   └── rectified_flow.py   # RectifiedFlow (copied from cosmos)
│   └── utils/
├── configs/
│   └── default.py           # Training configuration
├── scripts/
│   └── train.sh             # Training launcher
├── train.py                 # Main training script
└── ARCHITECTURE.md          # This file
```

## Key Components

### 1. VideoDiT (`src/cosmost/models/dit.py`)

Simplified DiT for video generation.

**Key features:**
- Patch embedding: Conv3d for video latents
- Timestep embedding: Sinusoidal + MLP
- Cross-attention: Supports text + action conditioning
- Action projection: Projects action sequence to context

**Forward flow:**
```
Video [B,16,12,60,80]
  ↓ Patch Embed
Tokens [B, T*H*W, D]
  ↓ + Timestep emb + Cross-attn (Text + Action)
DiT Blocks x28
  ↓
Output [B,16,12,60,80]
```

### 2. ActionHead (`src/cosmost/models/action_head.py`)

Transformer-based action prediction head.

**Key features:**
- Causal self-attention for actions
- Cross-attention to video features
- Layer-aligned with DiT (28 blocks)
- Timestep + positional embeddings

**Forward flow:**
```
Noisy Actions [B,64,16] + Video Features [B, N, D]
  ↓
Action Transformer Blocks x28
  ↓
Predicted Velocity [B,64,16]
```

### 3. JointVideoActionModel (`src/cosmost/models/joint_model.py`)

Main model combining video and action generation.

**Training process:**
```python
# 1. Sample timesteps
t_video, t_action = sample_timesteps(batch_size)

# 2. Add noise
xt_video = add_noise(x0_video, t_video)
xt_action = add_noise(x0_action, t_action)

# 3. Video forward (conditioned on GT action)
pred_v_video, hidden = video_dit(xt_video, t_video, action=gt_action)
video_loss = mse(pred_v_video, target_v_video)

# 4. Action forward (conditioned on video cond frames)
video_cond = extract_cond_frames(hidden, num_cond=4)
pred_v_action = action_head(xt_action, video_cond, t_action)
action_loss = mse(pred_v_action, target_v_action)

# 5. Joint loss
total_loss = video_loss + action_loss_weight * action_loss
```

## Comparison with FastWAM

| Aspect | FastWAM | CosmosT (This) |
|--------|---------|----------------|
| Video-Action Interaction | MoT (mixed attention) | Video→Action via features |
| Action→Video | Action as context tokens | Action in timestep embedding |
| Training GT usage | GT action conditions video | GT action conditions video |
| Inference | Joint denoising | Sequential (can add joint later) |
| Architecture | Wan2.2 5B based | Cosmos 2B based |

## Design Decisions

### 1. Why separate ActionHead instead of MoT?

**MoT (FastWAM):**
- Video and action tokens attend to each other at every layer
- Strong interaction but more complex
- Requires matching architecture between experts

**ActionHead (This):**
- Simpler implementation
- Video features guide action prediction
- Easier to debug and modify

### 2. Why action in timestep instead of cross-attention?

**Current design (timestep):**
- Simpler to implement
- Less memory overhead
- Sufficient for initial experiments

**Alternative (cross-attention):**
- Stronger conditioning
- Similar to FastWAM
- Can be added later if needed

## Training

### Single GPU
```bash
export DATASET_ROOT=/path/to/lerobot
export LATENT_ROOT=/path/to/latents
python train.py --config configs/default.py --output_dir ./outputs
```

### Multi-GPU
```bash
export NPROC=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/train.sh
```

## Extending

### Add cross-attention conditioning for action

Modify `VideoDiT.forward()` to concatenate action tokens to context:

```python
# In VideoDiT.forward()
context = self.crossattn_proj(crossattn_emb)  # [B, L, D]

if action is not None:
    action_tokens = self.action_token_proj(action)  # [B, 64, D]
    context = torch.cat([context, action_tokens], dim=1)
```

### Add joint inference

Modify `JointVideoActionModel.sample()` to:
1. Initialize both noises
2. Alternately denoise video and action
3. Use current action to condition video

```python
for step in range(num_steps):
    # Video denoise with current action
    v_video = video_dit(latents_video, t, action=latents_action)
    latents_video = step(latents_video, v_video, dt)
    
    # Action denoise with video features
    v_action = action_head(latents_action, video_features, t)
    latents_action = step(latents_action, v_action, dt)
```

## Dependencies

- torch >= 2.0
- einops
- tensorboard
- (optional) wandb

No dependency on cosmos-predict2.5!
