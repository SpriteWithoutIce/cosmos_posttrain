# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# 自定义 Action-Conditioned 实验配置（纯 Video 训练，无 Action Expert）
# 继承自官方 action-conditioned 体系，使用 LeRobot latent dataset
#
# 数据集:
#   - LeRobot metadata: {LEROBOT_ROOT}/{task_name}/
#   - 预计算 latents: {LATENT_ROOT}/{task_name}/traj_{episode:06d}.pt
#
# 核心改动:
#   1. 使用标准 Text2WorldModelRectifiedFlow，不加 action expert
#   2. 自定义 Dataset：LeRobotLatentDataset，返回 latents + text_emb
#   3. 使用预计算 latent 直通模型（不做 VAE encode）

import os
import time
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
from einops import rearrange
from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import Video2WorldModelRectifiedFlowConfig
from models.precomputed_latent import IdentityLatentTokenizer, PrecomputedLatentVideo2WorldModel

# =============================================================================
# 0. 全局配置
# =============================================================================
LEROBOT_ROOT = os.environ.get(
    "LEROBOT_ROOT",
    "/home/jwhe/linyihan/datasets/lerobot_robotwin_eef_clean_50"
)
LATENT_ROOT = os.environ.get(
    "LATENT_ROOT",
    "/home/jwhe/linyihan/datasets/lerobot_latents"
)
COSMOS_TOKENIZER = os.environ.get(
    "COSMOS_TOKENIZER",
    "/home/jwhe/linyihan/cosmos/tokenizer.pth",
)
OPEN_LOOP_SAMPLE_EVERY = int(os.environ.get("OPEN_LOOP_SAMPLE_EVERY", "200"))
OPEN_LOOP_NUM_SAMPLES = int(os.environ.get("OPEN_LOOP_NUM_SAMPLES", "1"))
OPEN_LOOP_GUIDANCE = float(os.environ.get("OPEN_LOOP_GUIDANCE", "0.0"))
ACTION_HEAD_ENABLED = int(os.environ.get("ACTION_HEAD_ENABLED", "1"))
ACTION_HEAD_LOSS_WEIGHT = float(os.environ.get("ACTION_HEAD_LOSS_WEIGHT", "1.0"))
ACTION_HEAD_SAVE_EVERY = int(os.environ.get("ACTION_HEAD_SAVE_EVERY", "500"))
ACTION_HEAD_SAVE_DIR = os.environ.get(
    "ACTION_HEAD_SAVE_DIR",
    "/home/jwhe/linyihan/robot_posttrain/action_head_ckpt",
)
ACTION_HEAD_LOAD_PATH = os.environ.get("ACTION_HEAD_LOAD_PATH", "")

# ★ 你的 post-train checkpoint
PT_CKPT = os.environ.get(
    "COSMOS_PT_CKPT",
    "/home/jwhe/linyihan/cosmos/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"
)

ACTION_DIM = 16                     # 来自 info.json state 维度
NUM_FRAMES = 45                     # 每次采样的 raw video 帧数
NUM_LATENTS = 12                    # 每个 trajectory 的 latent 数量
STATE_T = NUM_LATENTS               # 直接训练预计算 latent，时长应与输入 latent 帧数一致
LATENT_STRIDE = 4                  # raw fps 50 → latent fps 12.5

EXPERIMENT_NAME = "my_video_experiment"


# =============================================================================
# 1. 自定义 Dataset：LeRobotLatentDataset（适配新的 latent 路径结构）
# =============================================================================

def _get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


class LeRobotLatentDataset(torch.utils.data.Dataset):
    """
    读取 LeRobot 格式的 latent 数据（latent 预计算在单独目录）。

    LeRobot metadata: {lerobot_root}/{task_name}/
    预计算 latents: {latent_root}/{task_name}/traj_{episode:06d}.pt

    数据采样逻辑:
      - 每个 trajectory 有 n 个 latent（从 .pt 文件的 latent_num_frames 确定）
      - 预测 latent idx: [1, n-8]，共 n-8 个样本
      - 每个样本: 4 条件 + 8 预测，共 12 个 latent
      - 条件不足 4 个时，用第 0 个 latent 重复补齐
      - action: 8 预测 latent * 8 action/latent = 64 个 action

    输出:
      - latents: (12, C, H, W) video latent tensor
      - actions: (64, action_dim)
      - text_emb: (seq_len, hidden_dim)
      - task_text: str
    """

    def __init__(
        self,
        lerobot_root: str,
        latent_root: str,
        time_division_factor: int = 4,
        num_cond_frames: int = 4,
        num_pred_frames: int = 8,
        num_actions_per_latent: int = 8,
        action_dim: int = 16,
        data_split: str = "train",
    ):
        self.lerobot_root = Path(lerobot_root)
        self.latent_root = Path(latent_root)
        self.time_division_factor = time_division_factor
        self.num_cond_frames = num_cond_frames
        self.num_pred_frames = num_pred_frames
        self.num_actions_per_latent = num_actions_per_latent
        self.action_dim = action_dim
        self.data_split = data_split

        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.datasets.utils import get_episode_data_index

        self.meta = LeRobotDatasetMetadata(
            str(self.lerobot_root), str(self.lerobot_root), revision="v2.1", force_cache_sync=False
        )
        self.episode_data_index = get_episode_data_index(
            self.meta.episodes, self.meta.episodes
        )

        self.episodes = self._parse_episodes()
        self._build_sample_index()
        # Lightweight LRU caches to avoid repeatedly parsing the same files.
        self._parquet_cache = OrderedDict()
        self._latent_cache = OrderedDict()
        self._parquet_cache_size = 16
        self._latent_cache_size = 16
        self._debug_interval_sec = float(os.environ.get("LATENT_DATASET_DEBUG_INTERVAL", "0"))
        self._last_debug_ts = time.time()
        self._sample_counter = 0

    @property
    def episodes(self):
        return getattr(self, "_episodes", None)

    @episodes.setter
    def episodes(self, value):
        self._episodes = value

    def _parse_episodes(self):
        episode_list = []
        for ep_idx, ep_info in self.meta.episodes.items():
            split = ep_info.get("split", "train")
            # print(f"Episode {ep_idx}: split={split}, data_split={self.data_split}")
            if split != self.data_split and self.data_split != "all":
                continue
            episode_list.append({"episode_index": ep_idx})
        return episode_list

    def _build_sample_index(self):
        self.sample_index = []
        self.episode_cumsum = [0]
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

        if episode_index in self._parquet_cache:
            self._parquet_cache.move_to_end(episode_index)
            actions_all, states_all = self._parquet_cache[episode_index]
        else:
            chunk = self.meta.get_episode_chunk(episode_index)
            parquet_path = (
                self.lerobot_root
                / f"data/chunk-{chunk:03d}"
                / f"episode_{episode_index:06d}.parquet"
            )
            df = pd.read_parquet(parquet_path, columns=["action", "observation.state"])
            actions_all = np.stack(df["action"].to_numpy()).astype(np.float32, copy=False)
            states_all = np.stack(df["observation.state"].to_numpy()).astype(np.float32, copy=False)
            self._parquet_cache[episode_index] = (actions_all, states_all)
            if len(self._parquet_cache) > self._parquet_cache_size:
                self._parquet_cache.popitem(last=False)

        actions = actions_all[start:end]
        states = states_all[start:end]
        return actions, states

    def _load_latents_and_metadata(self, episode_index: int):
        """从 .pt 文件加载完整的 trajectory latent 和元数据。"""
        if episode_index in self._latent_cache:
            self._latent_cache.move_to_end(episode_index)
            latents, frame_ids, text_emb, task_text, n_latents = self._latent_cache[episode_index]
            return latents, frame_ids, text_emb, task_text, n_latents

        task_name = self.lerobot_root.name
        latent_file = self.latent_root / task_name / f"traj_{episode_index:03d}.pt"
        if not latent_file.exists():
            raise FileNotFoundError(f"Missing latent file: {latent_file}")

        data = torch.load(latent_file, weights_only=False)
        latents = data["latent"].float()
        # Normalize latent layout to (T, C, H, W).
        # Some preprocess pipelines save (C, T, H, W), while this dataset logic expects time-first.
        if latents.ndim != 4:
            raise ValueError(f"Invalid latent shape {tuple(latents.shape)} for episode {episode_index}")
        if latents.shape[1] == self.action_dim:
            # already (T, C, H, W) for C=16
            pass
        elif latents.shape[0] == self.action_dim:
            # convert (C, T, H, W) -> (T, C, H, W)
            latents = rearrange(latents, "c t h w -> t c h w").contiguous()
        else:
            raise ValueError(
                f"Unrecognized latent layout {tuple(latents.shape)} for episode {episode_index}; "
                f"expected channel dim == {self.action_dim}"
            )
        n_latents = int(data["latent_num_frames"])
        frame_ids = data["frame_ids"]
        text_emb = data.get("text_emb", None)
        task_text = data.get("task_text", "")

        self._latent_cache[episode_index] = (latents, frame_ids, text_emb, task_text, n_latents)
        if len(self._latent_cache) > self._latent_cache_size:
            self._latent_cache.popitem(last=False)

        return latents, frame_ids, text_emb, task_text, n_latents

    def _get_n_latents_for_episode(self, episode_index: int) -> int:
        """从 .pt 文件获取指定 episode 的真实 latent 数量。"""
        task_name = self.lerobot_root.name
        latent_file = self.latent_root / task_name / f"traj_{episode_index:03d}.pt"
        if latent_file.exists():
            data = torch.load(latent_file, weights_only=False)
            return int(data["latent_num_frames"])
        return 0

    def __getitem__(self, idx: int):
        episode_index, pred_idx = self.sample_index[idx]
        self._sample_counter += 1

        all_latents, frame_ids, text_emb, task_text, n_latents = self._load_latents_and_metadata(
            episode_index
        )
        if text_emb is None:
            raise ValueError(
                f"Missing text_emb in latent file for episode {episode_index}. "
                "This training setup requires precomputed text embeddings."
            )
        frame_ids_t = None
        if frame_ids is not None:
            frame_ids_t = frame_ids if torch.is_tensor(frame_ids) else torch.as_tensor(frame_ids)

        # === 4 个条件 latent ===
        if pred_idx >= self.num_cond_frames:
            cond_latents = all_latents[pred_idx - self.num_cond_frames:pred_idx]
        else:
            actual_latents = all_latents[0:pred_idx]  # 例如 pred_idx=2 -> [0,1]
            num_pad = self.num_cond_frames - actual_latents.shape[0]
            pad_src = actual_latents[-1:] if actual_latents.shape[0] > 0 else all_latents[0:1]
            pad_latents = pad_src.repeat(num_pad, 1, 1, 1)
            # 不足4帧时，用“已有前置里的最后一个”补齐: 1->0000, 2->0111
            cond_latents = torch.cat([actual_latents, pad_latents], dim=0)

        # === 8 个预测 latent ===
        target_latents = all_latents[pred_idx:pred_idx + self.num_pred_frames]

        # === 合并: 4 + 8 = 12 ===
        final_latents = torch.cat([cond_latents, target_latents], dim=0)

        # === 64 个 action（按 pred_idx 对齐）===
        # 用户定义规则:
        #   action_start = (pred_idx - 1) * 4
        #   actions = [action_start : action_start + 8*8)
        #   state = state[action_start] (16-dim)
        num_actions = self.num_pred_frames * self.num_actions_per_latent
        action_start = (pred_idx - 1) * self.time_division_factor
        action_end = action_start + num_actions
        actions, states_seq = self._load_parquet(episode_index, action_start, action_end)
        state_current = states_seq[0]

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

        if self._debug_interval_sec > 0:
            now = time.time()
            if now - self._last_debug_ts >= self._debug_interval_sec:
                print(
                    f"[latent-dset] pid={os.getpid()} samples={self._sample_counter} "
                    f"idx={idx} ep={episode_index} pred_idx={pred_idx} "
                    f"latent_cache={len(self._latent_cache)} parquet_cache={len(self._parquet_cache)}",
                    flush=True,
                )
                self._last_debug_ts = now

        # Model expects video latent shape as (C, T, H, W) per sample.
        video_cthw = rearrange(final_latents, "t c h w -> c t h w").contiguous()

        return {
            "video": video_cthw,                 # 模型读 "video"
            "actions": torch.from_numpy(actions).float(),
            "states": torch.from_numpy(state_current).float(),
            "states_seq": torch.from_numpy(states_seq).float(),
            "episode_index": episode_index,
            "pred_idx": pred_idx,
            "t5_text_embeddings": text_emb,      # 模型读 "t5_text_embeddings"
            "ai_caption": task_text,             # 模型读 "ai_caption"
            "frame_ids": sample_frame_ids,
            "fps": torch.tensor(16, dtype=torch.int32),
            "padding_mask": torch.zeros((1, final_latents.shape[-2], final_latents.shape[-1]), dtype=torch.float32),
        }

    def __len__(self):
        return len(self.sample_index)


class MultiLeRobotLatentDataset(torch.utils.data.Dataset):
    """
    支持多任务数据集目录，每个子 Dataset 已展平为样本数。
    """

    def __init__(
        self,
        lerobot_root: str,
        latent_root: str,
        time_division_factor: int = 4,
        num_cond_frames: int = 4,
        num_pred_frames: int = 8,
        num_actions_per_latent: int = 8,
        action_dim: int = 16,
        data_split: str = "train",
    ):
        self.datasets = []
        self.acc_offsets = [0]

        lerobot_path = Path(lerobot_root)
        for task_dir in sorted(lerobot_path.iterdir()):
            if not task_dir.is_dir() or task_dir.name.startswith("."):
                continue
            try:
                dset = LeRobotLatentDataset(
                    lerobot_root=str(task_dir),
                    latent_root=latent_root,
                    time_division_factor=time_division_factor,
                    num_cond_frames=num_cond_frames,
                    num_pred_frames=num_pred_frames,
                    num_actions_per_latent=num_actions_per_latent,
                    action_dim=action_dim,
                    data_split=data_split,
                )
                if len(dset) == 0:
                    continue
                self.datasets.append(dset)
            except Exception:
                continue

        for dset in self.datasets:
            self.acc_offsets.append(self.acc_offsets[-1] + len(dset))

    def __len__(self):
        return self.acc_offsets[-1]

    def __getitem__(self, idx: int):
        dset_idx = 0
        for i, offset in enumerate(self.acc_offsets[:-1]):
            if idx < self.acc_offsets[i + 1]:
                dset_idx = i
                break
        local_idx = idx - self.acc_offsets[dset_idx]
        return self.datasets[dset_idx][local_idx]


class CompatibleDataLoader(DataLoader):
    """
    Some inherited Cosmos configs inject fields for mixed-dataloader training
    (e.g. `dataloaders`, `ratio`) into `dataloader_train`.
    This wrapper ignores those extra fields and builds a normal PyTorch DataLoader.
    """

    def __init__(self, *args, dataloaders=None, ratio=None, **kwargs):
        del dataloaders, ratio
        super().__init__(*args, **kwargs)


# =============================================================================
# 2. Experiment Config（注册 DataLoader）
# =============================================================================

PRECOMPUTED_LATENT_FSDP_RECTIFIED_FLOW_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(PrecomputedLatentVideo2WorldModel)(
        action_head_enabled=bool(ACTION_HEAD_ENABLED),
        action_loss_weight=ACTION_HEAD_LOSS_WEIGHT,
        action_head_save_every=ACTION_HEAD_SAVE_EVERY,
        action_head_save_dir=ACTION_HEAD_SAVE_DIR,
        action_head_load_path=ACTION_HEAD_LOAD_PATH if ACTION_HEAD_LOAD_PATH else None,
        action_head_cfg=dict(
            action_dim=16,
            num_actions=64,
            d_model=256,
            n_heads=8,
            n_blocks=8,
            delta_dim=16,
            state_dim=16,
            sigma_k=4.0,
            dropout=0.0,
        ),
        config=Video2WorldModelRectifiedFlowConfig(
            fsdp_shard_size=2,
            state_t=STATE_T,
            text_encoder_config=None,  # 使用预计算 text_emb，不在线加载 reason1
            tokenizer=L(IdentityLatentTokenizer)(
                latent_ch=16,
                spatial_compression_factor=8,
                enable_decode=True,
                vae_pth=COSMOS_TOKENIZER,
                temporal_window=16,
            ),
        ),
        _recursive_=False,
    ),
)


my_video_experiment = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only",
            {"override /model": "precomputed_latent_video2world_fsdp_rectified_flow"},
            {"override /net": "cosmos_v1_2B"},
            {"override /conditioner": "video_prediction_conditioner"},
            {"override /data_train": "lerobot_eef_50_train"},
            {"override /data_val": "lerobot_eef_50_val"},
            "_self_",
        ],

        job=dict(
            project="cosmos_diffusion_v2",
            group="robot_posttrain",
            name="my_video_experiment",
        ),

        checkpoint=dict(
            save_iter=500,
            load_path=PT_CKPT,
            load_training_state=False,
            strict_resume=True,
            load_from_object_store=dict(enabled=False),
            save_to_object_store=dict(enabled=False),
        ),

        optimizer=dict(
            lr=2e-5,
            weight_decay=0.01,
        ),

        trainer=dict(
            straggler_detection=dict(enabled=False),
            max_iter=10_000,
            logging_iter=50,
            validation_iter=500,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=OPEN_LOOP_SAMPLE_EVERY,
                    do_x0_prediction=False,
                    guidance=[OPEN_LOOP_GUIDANCE for _ in range(OPEN_LOOP_NUM_SAMPLES)],
                    fps=16,
                    save_s3=False,
                ),
                every_n_sample_ema=dict(
                    every_n=OPEN_LOOP_SAMPLE_EVERY,
                    do_x0_prediction=False,
                    guidance=[OPEN_LOOP_GUIDANCE for _ in range(OPEN_LOOP_NUM_SAMPLES)],
                    fps=16,
                    save_s3=False,
                ),
                heart_beat=dict(save_s3=False),
                iter_speed=dict(hit_thres=100, save_s3=False),
                device_monitor=dict(save_s3=False),
                wandb=dict(save_s3=False),
                wandb_10x=dict(save_s3=False),
                dataloader_speed=dict(save_s3=False),
            ),
        ),

        model_parallel=dict(
            context_parallel_size=1,
        ),

        # 使用标准 Text2WorldModelRectifiedFlow（无 action expert）
        model=dict(
            config=dict(
                # 4 个条件 latent + 8 个预测 latent
                min_num_conditional_frames=4,
                max_num_conditional_frames=4,
                conditional_frames_probs=None,
                state_t=STATE_T,
                text_encoder_config=None,
            ),
        ),

    ),
    flags={"allow_objects": True},
)


# =============================================================================
# 3. DataLoader 注册
# =============================================================================

_lerobot_train_dataset = L(MultiLeRobotLatentDataset)(
    lerobot_root=LEROBOT_ROOT,
    latent_root=LATENT_ROOT,
    time_division_factor=LATENT_STRIDE,
    num_cond_frames=4,
    num_pred_frames=8,
    num_actions_per_latent=8,
    action_dim=ACTION_DIM,
    data_split="train",
)

_lerobot_val_dataset = L(MultiLeRobotLatentDataset)(
    lerobot_root=LEROBOT_ROOT,
    latent_root=LATENT_ROOT,
    time_division_factor=LATENT_STRIDE,
    num_cond_frames=4,
    num_pred_frames=8,
    num_actions_per_latent=8,
    action_dim=ACTION_DIM,
    data_split="test",
)

lerobot_eef_50_train_dataloader = L(CompatibleDataLoader)(
    dataset=_lerobot_train_dataset,
    sampler=L(_get_sampler)(dataset=_lerobot_train_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=0,
    pin_memory=True,
)

lerobot_eef_50_val_dataloader = L(CompatibleDataLoader)(
    dataset=_lerobot_val_dataset,
    sampler=L(_get_sampler)(dataset=_lerobot_val_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=0,
    pin_memory=True,
)


def register_lerobot_eef_data():
    cs = ConfigStore.instance()
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="lerobot_eef_50_train",
        node=lerobot_eef_50_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="lerobot_eef_50_val",
        node=lerobot_eef_50_val_dataloader,
    )


# =============================================================================
# 4. Hydra 注册
# =============================================================================
cs = ConfigStore.instance()

# 注册预计算 latent 专用模型
cs.store(
    group="model",
    package="_global_",
    name="precomputed_latent_video2world_fsdp_rectified_flow",
    node=PRECOMPUTED_LATENT_FSDP_RECTIFIED_FLOW_CONFIG,
)

# 注册 Experiment
cs.store(
    group="experiment",
    package="_global_",
    name=EXPERIMENT_NAME,
    node=my_video_experiment,
)

# 注册 DataLoader
register_lerobot_eef_data()
