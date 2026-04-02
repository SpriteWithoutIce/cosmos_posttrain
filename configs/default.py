"""
Default configuration for Joint Video-Action Training.
"""

import os


def get_config():
    """Returns training configuration."""
    
    # Paths (update these to your actual paths)
    base_dir = os.environ.get("DATASET_ROOT", "/home/jwhe/linyihan/datasets/lerobot_robotwin_eef_clean_50")
    latent_dir = os.environ.get("LATENT_ROOT", "/home/jwhe/linyihan/datasets/lerobot_latents")
    
    return {
        # Data
        "lerobot_root": base_dir,
        "latent_root": latent_dir,
        "global_stats_json": os.path.join(base_dir, "stats.json"),
        "num_cond_frames": 4,
        "num_pred_frames": 8,
        "num_actions_per_frame": 8,
        "time_division_factor": 8,  # 图像降采样后，一个 latent 对应 8 个 action
        "action_dim": 16,
        "normalize_action": True,
        
        # Video DiT
        "video_dit": {
            "in_channels": 16,
            "out_channels": 16,
            "model_channels": 2048,
            "num_blocks": 28,
            "num_heads": 16,
            "mlp_ratio": 4.0,
            "crossattn_dim": 100352,  # T5 embedding dim (matching your text embeddings)
            "patch_spatial": 2,
            "patch_temporal": 1,
            "max_frames": 128,
            "max_height": 240,
            "max_width": 240,
            "use_adaln_lora": True,
            "adaln_lora_dim": 256,
        },
        
        # Action Head
        "action_head_enabled": True,
        "action_head": {
            "action_dim": 16,
            "num_actions": 64,
            "d_model": 1024,
            "n_heads": 8,
            "video_hidden_dim": 2048,
            "dropout": 0.0,
        },
        "action_loss_weight": 1.0,
        "action_stop_gradient": False,  # Must be False for joint training
        
        # Training
        "batch_size": 1,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "max_iterations": 100000,
        "save_every": 5000,
        "log_every": 100,
        "num_workers": 4,
    }
