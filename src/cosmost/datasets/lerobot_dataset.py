"""
LeRobot Latent Dataset for Video-Action Joint Training.
Simplified from the original implementation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class LeRobotLatentDataset(Dataset):
    """
    Dataset for precomputed latents with actions.
    
    Data structure:
    - LeRobot metadata: {lerobot_root}/{task_name}/
    - Precomputed latents: {latent_root}/{task_name}/traj_{episode:06d}.pt
    
    Each sample contains:
    - video: [C, T, H, W] latent tensor
    - actions: [64, action_dim] action tensor
    - text_emb: [L, D] text embedding
    """
    
    def __init__(
        self,
        lerobot_root: str,
        latent_root: str,
        num_cond_frames: int = 4,
        num_pred_frames: int = 8,
        num_actions_per_frame: int = 8,
        action_dim: int = 16,
        data_split: str = "train",
        normalize_action: bool = True,
        global_stats_json: Optional[str] = None,
    ):
        self.lerobot_root = Path(lerobot_root)
        self.latent_root = Path(latent_root)
        self.num_cond_frames = num_cond_frames
        self.num_pred_frames = num_pred_frames
        self.num_actions_per_frame = num_actions_per_frame
        self.action_dim = action_dim
        self.data_split = data_split
        self.normalize_action = normalize_action
        
        # Load LeRobot metadata
        self._load_metadata()
        
        # Build sample index
        self._build_sample_index()
        
        # Load normalization stats
        self._load_stats(global_stats_json)
    
    def _load_metadata(self):
        """Load LeRobot dataset metadata."""
        # Simplified: assume we have info.json
        info_path = self.lerobot_root / "info.json"
        if info_path.exists():
            with open(info_path) as f:
                self.info = json.load(f)
        else:
            self.info = {}
        
        # Load episodes
        self.episodes = []
        episodes_dir = self.latent_root
        
        print(f"[Dataset] Looking for latents in: {episodes_dir}")
        print(f"[Dataset] Absolute path: {episodes_dir.absolute()}")
        
        if not episodes_dir.exists():
            print(f"[Dataset] WARNING: Latent directory does not exist: {episodes_dir}")
            print(f"[Dataset] Current working directory: {os.getcwd()}")
            print(f"[Dataset] Parent exists: {episodes_dir.parent.exists()}")
            if episodes_dir.parent.exists():
                print(f"[Dataset] Parent contents: {list(episodes_dir.parent.iterdir())}")
            return
            
        traj_files = list(episodes_dir.glob("traj_*.pt"))
        print(f"[Dataset] Found {len(traj_files)} traj files")
        
        # Show first few files
        if traj_files:
            print(f"[Dataset] First 5 files: {[f.name for f in traj_files[:5]]}")
        
        for traj_file in sorted(traj_files):
            try:
                episode_idx = int(traj_file.stem.split("_")[1])
                self.episodes.append(episode_idx)
            except (ValueError, IndexError) as e:
                print(f"[Dataset] Warning: Could not parse episode index from {traj_file}: {e}")
                continue
        
        print(f"[Dataset] Loaded {len(self.episodes)} episodes: {self.episodes[:10]}...")
    
    def _build_sample_index(self):
        """Build index of valid samples."""
        self.samples = []
        
        print(f"[Dataset] Building sample index for {len(self.episodes)} episodes...")
        
        for episode_idx in self.episodes:
            latent_file = self.latent_root / f"traj_{episode_idx:06d}.pt"
            if not latent_file.exists():
                # Try without leading zeros
                latent_file_alt = self.latent_root / f"traj_{episode_idx}.pt"
                if latent_file_alt.exists():
                    latent_file = latent_file_alt
                else:
                    # Try with 3 digits
                    latent_file_alt2 = self.latent_root / f"traj_{episode_idx:03d}.pt"
                    if latent_file_alt2.exists():
                        latent_file = latent_file_alt2
                    else:
                        print(f"[Dataset] Warning: Latent file not found: {latent_file}")
                        continue
            
            # Load to get number of frames
            try:
                data = torch.load(latent_file, weights_only=False)
                n_latents = int(data.get("latent_num_frames", data["latent"].shape[0]))
            except Exception as e:
                print(f"[Dataset] Warning: Failed to load {latent_file}: {e}")
                continue
            
            # Create samples: need at least num_cond_frames + num_pred_frames
            min_frames = self.num_cond_frames + self.num_pred_frames
            if n_latents < min_frames:
                print(f"[Dataset] Warning: Episode {episode_idx} has only {n_latents} frames, need {min_frames}")
                continue
            
            # Create samples: pred_idx from 1 to n_latents - num_pred_frames
            for pred_idx in range(1, n_latents - self.num_pred_frames + 1):
                self.samples.append((episode_idx, pred_idx))
        
        print(f"[Dataset] Total samples: {len(self.samples)}")
        if len(self.samples) == 0:
            print(f"[Dataset] WARNING: No valid samples found!")
    
    def _load_stats(self, global_stats_json: Optional[str]):
        """Load action normalization statistics."""
        self.action_q01 = None
        self.action_q99 = None
        self.state_q01 = None
        self.state_q99 = None
        
        if not self.normalize_action:
            return
        
        stats_path = global_stats_json or self.lerobot_root / "stats.json"
        if not Path(stats_path).exists():
            return
        
        try:
            with open(stats_path) as f:
                stats = json.load(f)
            
            action_stats = stats.get("action", stats.get("actions"))
            state_stats = stats.get("observation.state", stats.get("state"))
            
            if action_stats and "q01" in action_stats:
                self.action_q01 = torch.tensor(action_stats["q01"], dtype=torch.float32)
                self.action_q99 = torch.tensor(action_stats["q99"], dtype=torch.float32)
            
            if state_stats and "q01" in state_stats:
                self.state_q01 = torch.tensor(state_stats["q01"], dtype=torch.float32)
                self.state_q99 = torch.tensor(state_stats["q99"], dtype=torch.float32)
        except Exception as e:
            print(f"Warning: Failed to load stats: {e}")
    
    def _qnormalize(self, x: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
        """Quantile normalization to [-1, 1]."""
        eps = 1e-6
        q01 = q01.to(device=x.device, dtype=x.dtype)
        q99 = q99.to(device=x.device, dtype=x.dtype)
        return 2.0 * (x - q01) / (q99 - q01 + eps) - 1.0
    
    def _load_latent(self, episode_idx: int) -> Dict[str, torch.Tensor]:
        """Load latent file for an episode."""
        latent_file = self.latent_root / f"traj_{episode_idx:06d}.pt"
        data = torch.load(latent_file, weights_only=False)
        
        latents = data["latent"].float()
        
        # Ensure shape is [T, C, H, W]
        if latents.ndim == 4 and latents.shape[0] != latents.shape[1]:
            # Already [T, C, H, W]
            pass
        elif latents.ndim == 4 and latents.shape[1] == self.action_dim:
            # [C, T, H, W] -> [T, C, H, W]
            latents = latents.permute(1, 0, 2, 3)
        
        return {
            "latents": latents,
            "text_emb": data.get("text_emb"),
            "task_text": data.get("task_text", ""),
        }
    
    def _load_actions(self, episode_idx: int, start_idx: int, end_idx: int) -> torch.Tensor:
        """Load actions from parquet."""
        import pandas as pd
        
        # Find parquet file
        parquet_files = list(self.lerobot_root.glob(f"**/episode_{episode_idx:06d}.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet found for episode {episode_idx}")
        
        df = pd.read_parquet(parquet_files[0], columns=["action"])
        actions = np.stack(df["action"].to_numpy()).astype(np.float32)
        actions = torch.from_numpy(actions[start_idx:end_idx])
        
        # Normalize
        if self.normalize_action and self.action_q01 is not None:
            actions = self._qnormalize(actions, self.action_q01, self.action_q99)
        
        return actions
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        episode_idx, pred_idx = self.samples[idx]
        
        # Load latent
        latent_data = self._load_latent(episode_idx)
        all_latents = latent_data["latents"]  # [T, C, H, W]
        
        # Extract conditional latents
        if pred_idx >= self.num_cond_frames:
            cond_latents = all_latents[pred_idx - self.num_cond_frames:pred_idx]
        else:
            # Pad with first frame
            actual = all_latents[:pred_idx]
            if len(actual) > 0:
                pad = actual[-1:].repeat(self.num_cond_frames - len(actual), 1, 1, 1)
                cond_latents = torch.cat([actual, pad], dim=0)
            else:
                cond_latents = all_latents[0:1].repeat(self.num_cond_frames, 1, 1, 1)
        
        # Extract prediction latents
        pred_latents = all_latents[pred_idx:pred_idx + self.num_pred_frames]
        
        # Combine: [num_cond + num_pred, C, H, W]
        video = torch.cat([cond_latents, pred_latents], dim=0)
        
        # Rearrange to [C, T, H, W]
        video = video.permute(1, 0, 2, 3).contiguous()
        
        # Load actions
        num_actions = self.num_pred_frames * self.num_actions_per_frame
        action_start = (pred_idx - 1) * 4  # time_division_factor = 4
        action_end = action_start + num_actions
        actions = self._load_actions(episode_idx, action_start, action_end)
        
        # Get text embedding
        text_emb = latent_data["text_emb"]
        if text_emb is None:
            raise ValueError(f"Missing text_emb in episode {episode_idx}")
        
        return {
            "video": video,
            "actions": actions,
            "t5_text_embeddings": text_emb,
            "episode_index": episode_idx,
            "pred_idx": pred_idx,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function with padding for text embeddings."""
    keys = batch[0].keys()
    result = {}
    
    for key in keys:
        values = [item[key] for item in batch]
        
        if key == "t5_text_embeddings":
            # Pad text embeddings to max length
            max_len = max(v.shape[0] for v in values)
            hidden_dim = values[0].shape[-1]
            dtype = values[0].dtype
            
            padded = values[0].new_zeros((len(values), max_len, hidden_dim), dtype=dtype)
            mask = torch.zeros((len(values), max_len), dtype=torch.float32)
            
            for i, emb in enumerate(values):
                seq_len = emb.shape[0]
                padded[i, :seq_len] = emb
                mask[i, :seq_len] = 1.0
            
            result[key] = padded
            result["t5_text_mask"] = mask
        elif isinstance(values[0], str):
            result[key] = values
        else:
            result[key] = torch.stack(values, dim=0)
    
    return result
