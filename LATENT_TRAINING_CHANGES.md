# Latent-Only Training Changes

## Scope
All changes are in `cosmos_posttrain` only.
No files under `cosmos-predict2.5-main` were modified.

## What changed

1. Added latent-only model components:
- `models/precomputed_latent.py`
  - `IdentityLatentTokenizer`: satisfies tokenizer interface without loading VAE checkpoints.
  - `PrecomputedLatentVideo2WorldModel`: consumes precomputed latents directly and skips pixel normalization + VAE encode.
- `models/__init__.py`

2. Updated experiment config:
- `configs/experiments/my_action_experiment.py`
  - Added and registered model group: `precomputed_latent_video2world_fsdp_rectified_flow`.
  - Model now uses `PrecomputedLatentVideo2WorldModel` + `IdentityLatentTokenizer`.
  - Disabled online text encoding via `text_encoder_config=None`.
  - `STATE_T` changed from `1 + NUM_LATENTS // 4` to `NUM_LATENTS` for latent-direct training.
  - `PT_CKPT` now reads from env `COSMOS_PT_CKPT`.
  - Dataset output now includes required keys:
    - `fps`
    - `padding_mask`
  - Added strict check that each sample must contain precomputed `text_emb`.

3. Registered video net configs so `cosmos_v1_2B` is available:
- `configs/config.py`
  - Added `register_video_net()` call from video2world defaults.

4. Environment and launcher alignment:
- `export_env.sh`
  - Added:
    - `LEROBOT_ROOT` (defaults to `DATASET_ROOT`)
    - `LATENT_ROOT` (defaults to `/home/jwhe/linyihan/datasets/lerobot_latents`)
- `train.sh`
  - `PYTHONPATH` now uses configurable `COSMOS_PREDICT2_ROOT`.
  - Default path: `/home/jwhe/linyihan/cosmos-predict2.5-main`.

## Validation done
- `python3 -m py_compile models/precomputed_latent.py configs/config.py configs/experiments/my_action_experiment.py scripts/train.py`

