# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
#
# 适配 LeRobot v2.1 latent dataset → Cosmos-Predict2.5 action-conditioned 输入
# 不做归一化，直接输出原始 action / state 张量。
#
# 数据结构:
#   - LeRobot metadata: {lerobot_root}/{task_name}/meta/, videos/, data/
#   - 预计算 latents: {latent_root}/{task_name}/traj_{episode:06d}.pt
#     内容: latent (16, F, H, W), text_emb, task_text, frame_ids 等

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from einops import rearrange
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import get_episode_data_index
from torch.utils.data import Dataset


class LeRobotLatentDataset(Dataset):
    """
    读取 LeRobot 格式的 latent 数据（latent 预计算在单独目录）。

    LeRobot metadata: {lerobot_root}/{task_name}/
    预计算 latents: {latent_root}/{task_name}/traj_{episode:06d}.pt

    数据采样逻辑:
      - 每个 trajectory 有 n 个 latent（从 .pt 文件的 frame_ids 确定）
      - 预测 latent idx 范围: [1, n-8]，共 n-8 个样本
      - 每个样本: 4 个条件 latent + 8 个预测 latent，共 12 个
      - 条件不足 4 个时，用第 0 个 latent 重复补齐
      - action: 8 个预测 latent 对应的 8*8=64 个 action

    输出:
      - latents: (12, C, H, W)  video latent tensor (4条件 + 8预测)
      - actions: (64, action_dim) 原始 action 序列（8 pred latent * 8 action/latent）
      - text_emb: (seq_len, hidden_dim) text embedding
      - task_text: str
    """

    def __init__(
        self,
        lerobot_root: str,
        latent_root: str,
        time_division_factor: int = 4,
        num_cond_frames: int = 4,
        num_pred_frames: int = 8,
        num_actions_per_latent: int = 4,
        *,
        obs_cam_keys: Sequence[str] | None = None,
        action_dim: int = 16,
        data_split: str = "train",
        env_type: str = "robotwin_tshape",
    ):
        self.lerobot_root = Path(lerobot_root)
        self.latent_root = Path(latent_root)
        self.time_division_factor = time_division_factor
        self.num_cond_frames = num_cond_frames
        self.num_pred_frames = num_pred_frames
        self.num_actions_per_latent = num_actions_per_latent
        self.obs_cam_keys = obs_cam_keys or [
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ]
        self.action_dim = action_dim
        self.data_split = data_split
        self.env_type = env_type
        self.cfg_prob = 0.0

        self.meta = LeRobotDatasetMetadata(
            self.lerobot_root, self.lerobot_root, revision="v2.1", force_cache_sync=False
        )
        self.episode_data_index = get_episode_data_index(
            self.meta.episodes, self.meta.episodes
        )
        self.episodes = self._parse_episodes()
        self._build_sample_index()  # 预计算扁平化索引表

    @property
    def episodes(self):
        return getattr(self, "_episodes", None)

    @episodes.setter
    def episodes(self, value):
        self._episodes = value

    def _parse_episodes(self):
        """
        每个 trajectory 生成 (n - 8) 个样本，n 为该 traj 的 latent 总数。
        预测 idx: 1 到 n-8。
        注意: n 从 .pt 文件的 latent_num_frames 获取，不从 ep_len 估算。
        在 __init__ 时预计算每个 traj 的样本数和累积偏移量。
        """
        episode_list = []
        for ep_idx, ep_info in self.meta.episodes.items():
            split = ep_info.get("split", "train")
            if split != self.data_split and self.data_split != "all":
                continue
            episode_list.append({"episode_index": ep_idx})
        return episode_list

    def _build_sample_index(self):
        """
        预计算扁平化的样本索引表和累积偏移量。
        self.sample_index: list of (episode_index, pred_idx)
        self.episode_cumsum: cumsum of samples per episode
        """
        self.sample_index = []  # list of (episode_index, pred_idx)
        self.episode_cumsum = [0]  # cumulative number of samples

        for ep in self.episodes:
            ep_idx = ep["episode_index"]
            n_latents = self._get_n_latents_for_episode(ep_idx)
            n_samples = max(0, n_latents - self.num_pred_frames)  # pred_idx from 1 to n-8
            for pred_idx in range(1, n_latents - self.num_pred_frames + 1):
                self.sample_index.append((ep_idx, pred_idx))
            self.episode_cumsum.append(self.episode_cumsum[-1] + n_samples)

    def _load_parquet(self, episode_index: int, start: int, end: int):
        """从 parquet 加载 action / state。"""
        import pandas as pd

        chunk = self.meta.get_episode_chunk(episode_index)
        parquet_path = (
            self.lerobot_root
            / f"data/chunk-{chunk:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        df = pd.read_parquet(parquet_path)
        row = df.iloc[start:end]
        actions = row["action"].values.astype(np.float32)
        states = row["observation.state"].values.astype(np.float32)
        return actions, states

    def _load_latents_and_metadata(self, episode_index: int):
        """
        从 .pt 文件加载完整的 trajectory latent 和元数据。
        返回: (n_latents, C, H, W), frame_ids, text_emb, task_text, n_latents
        """
        task_name = self.lerobot_root.name
        latent_file = self.latent_root / task_name / f"traj_{episode_index:06d}.pt"

        data = torch.load(latent_file, weights_only=False)
        latents = data["latent"]  # (n_latents, C, H, W)
        n_latents = data["latent_num_frames"]  # 真实的 latent 数量（整数）
        frame_ids = data["frame_ids"]         # tensor of raw frame indices
        text_emb = data.get("text_emb", None)
        task_text = data.get("task_text", "")

        return latents.float(), frame_ids, text_emb, task_text, n_latents

    def __getitem__(self, idx: int) -> dict:
        # 用预计算的扁平化索引表直接查找
        episode_index, pred_idx = self.sample_index[idx]

        # 加载 .pt 获取 latent 和元数据
        all_latents, frame_ids, text_emb, task_text, n_latents = self._load_latents_and_metadata(
            episode_index
        )
        frame_ids_t = None
        if frame_ids is not None:
            frame_ids_t = frame_ids if torch.is_tensor(frame_ids) else torch.as_tensor(frame_ids)

        # === 构建 4 个条件 latent ===
        if pred_idx >= self.num_cond_frames:
            cond_latents = all_latents[pred_idx - self.num_cond_frames:pred_idx]
        else:
            actual_latents = all_latents[0:pred_idx]  # 例如 pred_idx=2 -> [0,1]
            num_pad = self.num_cond_frames - actual_latents.shape[0]
            pad_src = actual_latents[-1:] if actual_latents.shape[0] > 0 else all_latents[0:1]
            pad_latents = pad_src.repeat(num_pad, 1, 1, 1)
            # 不足4帧时，用“已有前置里的最后一个”补齐: 1->0000, 2->0111
            cond_latents = torch.cat([actual_latents, pad_latents], dim=0)

        # === 构建 8 个预测 latent ===
        target_latents = all_latents[pred_idx:pred_idx + self.num_pred_frames]

        # === 合并: 4 + 8 = 12 ===
        final_latents = torch.cat([cond_latents, target_latents], dim=0)

        # === 加载 64 个 action ===
        frame_stride = (
            frame_ids_t[1] - frame_ids_t[0]
            if frame_ids_t is not None and len(frame_ids_t) > 1
            else 1
        )
        num_actions = self.num_pred_frames * self.num_actions_per_latent * frame_stride
        if frame_ids_t is not None:
            action_start = int(frame_ids_t[pred_idx].item())
        else:
            action_start = pred_idx * self.time_division_factor
        action_end = action_start + num_actions
        actions, states = self._load_parquet(episode_index, action_start, action_end)

        sample_frame_ids = None
        if frame_ids_t is not None:
            if pred_idx >= self.num_cond_frames:
                cond_frame_ids = frame_ids_t[pred_idx - self.num_cond_frames:pred_idx]
            else:
                actual_frame_ids = frame_ids_t[0:pred_idx]
                num_pad = self.num_cond_frames - actual_frame_ids.shape[0]
                pad_src = (
                    actual_frame_ids[-1:]
                    if actual_frame_ids.shape[0] > 0
                    else frame_ids_t[0:1]
                )
                pad_frame_ids = pad_src.repeat(num_pad)
                cond_frame_ids = torch.cat([actual_frame_ids, pad_frame_ids], dim=0)
            target_frame_ids = frame_ids_t[pred_idx:pred_idx + self.num_pred_frames]
            sample_frame_ids = torch.cat([cond_frame_ids, target_frame_ids], dim=0)

        return {
            "video": final_latents,              # 模型读 "video"
            "actions": torch.from_numpy(actions).float(),
            "states": torch.from_numpy(states).float(),
            "episode_index": episode_index,
            "pred_idx": pred_idx,
            "t5_text_embeddings": text_emb,      # 模型读 "t5_text_embeddings"
            "ai_caption": task_text,             # 模型读 "ai_caption"
            "frame_ids": sample_frame_ids,
        }

    def _get_n_latents_for_episode(self, episode_index: int) -> int:
        """从 .pt 文件获取指定 episode 的真实 latent 数量。"""
        task_name = self.lerobot_root.name
        latent_file = self.latent_root / task_name / f"traj_{episode_index:06d}.pt"
        if latent_file.exists():
            data = torch.load(latent_file, weights_only=False)
            return int(data["latent_num_frames"])
        return 0

    def __len__(self) -> int:
        return len(self.sample_index)


class MultiLeRobotLatentDataset(torch.utils.data.Dataset):
    """
    支持多任务数据集目录（DATASET_ROOT 下有多个任务子文件夹），
    每个子文件夹独立加载 episode，统一索引。

    结构:
      - LeRobot metadata: {lerobot_root}/{task_name}/
      - 预计算 latents: {latent_root}/{task_name}/traj_XXX.pt

    子 LeRobotLatentDataset 已经展平为 (n - 8) 个样本，
    MultiLeRobotLatentDataset 把多个子 Dataset 的样本拼接起来。
    """

    def __init__(
        self,
        lerobot_root: str,
        latent_root: str,
        time_division_factor: int = 4,
        num_cond_frames: int = 4,
        num_pred_frames: int = 8,
        num_actions_per_latent: int = 4,
        *,
        obs_cam_keys: Sequence[str] | None = None,
        action_dim: int = 16,
        data_split: str = "train",
        env_type: str = "robotwin_tshape",
    ):
        self.datasets = []
        self.acc_offsets = [0]

        lerobot_root_path = Path(lerobot_root)
        latent_root_path = Path(latent_root)

        for task_dir in sorted(lerobot_root_path.iterdir()):
            if not task_dir.is_dir() or task_dir.name.startswith("."):
                continue
            try:
                dset = LeRobotLatentDataset(
                    lerobot_root=str(task_dir),
                    latent_root=str(latent_root_path),
                    time_division_factor=time_division_factor,
                    num_cond_frames=num_cond_frames,
                    num_pred_frames=num_pred_frames,
                    num_actions_per_latent=num_actions_per_latent,
                    obs_cam_keys=obs_cam_keys,
                    action_dim=action_dim,
                    data_split=data_split,
                    env_type=env_type,
                )
                if len(dset) == 0:
                    continue
                self.datasets.append(dset)
            except Exception:
                continue

        # 子 Dataset 已经展平为样本数，累加即可
        for dset in self.datasets:
            self.acc_offsets.append(self.acc_offsets[-1] + len(dset))

    def __len__(self) -> int:
        return self.acc_offsets[-1]

    def __getitem__(self, idx: int) -> dict:
        dset_idx = 0
        for i, offset in enumerate(self.acc_offsets[:-1]):
            if idx < self.acc_offsets[i + 1]:
                dset_idx = i
                break
        local_idx = idx - self.acc_offsets[dset_idx]
        return self.datasets[dset_idx][local_idx]
