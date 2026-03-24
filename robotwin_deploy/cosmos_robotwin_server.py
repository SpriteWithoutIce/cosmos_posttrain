from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from einops import rearrange

from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint
from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import WanVAE_

try:
    from .websocket_policy_server import WebsocketPolicyServer
except ImportError:
    from websocket_policy_server import WebsocketPolicyServer

logger = logging.getLogger(__name__)


@dataclass
class DeployConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "cuda"
    experiment_name: str = "my_video_experiment"
    config_file: str = "configs/config.py"
    video_ckpt: str = ""
    action_head_ckpt: str = ""
    vae_path: str = "/home/jwhe/linyihan/cosmos/tokenizer.pth"
    stats_json: str = ""
    num_cond_frames: int = 4
    num_pred_frames: int = 8
    output_frames: int = 4
    actions_per_frame: int = 8
    action_dim: int = 16
    latent_h: int = 60
    latent_w: int = 80
    guidance: float = 0.0
    num_steps: int = 10
    shift: float = 5.0
    seed: int = 1234
    text_emb_dim: int = 100352
    text_emb_pt: str = ""


class CosmosVAEWrapper(torch.nn.Module):
    def __init__(self, vae_pth: str, device: str = "cuda"):
        super().__init__()
        cfg = dict(
            dim=96,
            z_dim=16,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            temperal_downsample=[False, True, True],
            dropout=0.0,
            temporal_window=4,
        )
        with torch.device("meta"):
            self.model = WanVAE_(**cfg)
        ckpt = torch.load(vae_pth, map_location=device, weights_only=False)
        self.model.load_state_dict(ckpt, assign=True)
        self.model.eval().requires_grad_(False)
        self.model.to(device)

        mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
        std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
               3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
        self.register_buffer("_mean", torch.tensor(mean).reshape(1, 16, 1, 1, 1))
        self.register_buffer("_std", torch.tensor(std).reshape(1, 16, 1, 1, 1))

    @torch.no_grad()
    def encode(self, videos: torch.Tensor) -> torch.Tensor:
        in_dtype = videos.dtype
        device = videos.device
        self.model.to(device)
        scale = [self._mean.to(device), (1.0 / self._std).to(device)]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            latent = self.model.encode(videos.to(torch.bfloat16), scale)
        return latent.to(in_dtype)


class ActionStateQNorm:
    def __init__(self, stats_json: str):
        self.action_q01 = None
        self.action_q99 = None
        self.state_q01 = None
        self.state_q99 = None
        if not stats_json:
            return
        if not os.path.exists(stats_json):
            logger.warning("stats_json not found: %s", stats_json)
            return
        with open(stats_json, "r", encoding="utf-8") as f:
            stats = json.load(f)
        action_stats = stats.get("action") or stats.get("actions")
        state_stats = stats.get("observation.state") or stats.get("state") or stats.get("states")
        if action_stats and state_stats and "q01" in action_stats and "q99" in action_stats:
            self.action_q01 = np.asarray(action_stats["q01"], dtype=np.float32)
            self.action_q99 = np.asarray(action_stats["q99"], dtype=np.float32)
        if action_stats and state_stats and "q01" in state_stats and "q99" in state_stats:
            self.state_q01 = np.asarray(state_stats["q01"], dtype=np.float32)
            self.state_q99 = np.asarray(state_stats["q99"], dtype=np.float32)

    @staticmethod
    def _qnorm(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
        return 2.0 * (x - q01) / (q99 - q01 + 1e-6) - 1.0

    @staticmethod
    def _qunnorm(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
        return (x + 1.0) * 0.5 * (q99 - q01 + 1e-6) + q01

    def norm_state(self, state: np.ndarray) -> np.ndarray:
        if self.state_q01 is None or self.state_q99 is None:
            return state
        return self._qnorm(state, self.state_q01, self.state_q99).astype(np.float32)

    def unnorm_action(self, action: np.ndarray) -> np.ndarray:
        if self.action_q01 is None or self.action_q99 is None:
            return action
        return self._qunnorm(action, self.action_q01, self.action_q99).astype(np.float32)


class CosmosRobotWinServer:
    def __init__(self, cfg: DeployConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        logger.info("Loading tokenizer/vae from: %s", cfg.vae_path)
        self.vae = CosmosVAEWrapper(cfg.vae_path, device=str(self.device))
        logger.info("Loading model from video ckpt: %s", cfg.video_ckpt)
        self.model, _ = load_model_from_checkpoint(
            experiment_name=cfg.experiment_name,
            s3_checkpoint_dir=cfg.video_ckpt,
            config_file=cfg.config_file,
            enable_fsdp=False,
            load_ema_to_reg=False,
            to_device=str(self.device),
        )
        self.model.eval()
        if not hasattr(self.model, "action_head") or self.model.action_head is None:
            raise RuntimeError("Loaded model does not include action_head. Check experiment/config.")
        if cfg.action_head_ckpt:
            logger.info("Loading action head ckpt: %s", cfg.action_head_ckpt)
            self.model.load_action_head(cfg.action_head_ckpt, strict=True)
        self.qnorm = ActionStateQNorm(cfg.stats_json)
        self.text_emb_cache = self._load_text_embedding_cache(cfg.text_emb_pt)
        self._reset()

    def _reset(self):
        self.raw_obs_history: list[dict[str, np.ndarray]] = []
        self.latent_history: list[torch.Tensor] = []
        self.latest_state: np.ndarray | None = None
        self.step_id = 0

    def _load_text_embedding_cache(self, pkl_path: str) -> dict[str, torch.Tensor]:
        if not pkl_path:
            return {}
        if not os.path.exists(pkl_path):
            logger.warning("text_emb pkl not found: %s", pkl_path)
            return {}
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, dict):
            raise ValueError(f"text_emb pkl must contain dict, got {type(obj)}")
        out: dict[str, torch.Tensor] = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            if torch.is_tensor(v):
                t = v.detach().cpu()
            else:
                t = torch.as_tensor(v)
            if t.ndim == 2:
                t = t.unsqueeze(0)
            if t.ndim != 3:
                continue
            out[k] = t
        logger.info("Loaded %d task text embeddings from %s", len(out), pkl_path)
        return out

    def _get_text_embedding(
        self,
        task_name: str | None,
        prompt: str | None,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        # exact match: task_name first, then prompt
        for key in [task_name, prompt]:
            if key and key in self.text_emb_cache:
                emb = self.text_emb_cache[key]
                if emb.ndim == 2:
                    emb = emb.unsqueeze(0)
                return emb.to(device=self.device, dtype=target_dtype)
        # fuzzy match
        for lookup in [task_name, prompt]:
            if not lookup:
                continue
            lower_lookup = lookup.lower()
            for key, emb in self.text_emb_cache.items():
                lower_key = key.lower()
                if lower_lookup in lower_key or lower_key in lower_lookup:
                    if emb.ndim == 2:
                        emb = emb.unsqueeze(0)
                    logger.info("Fuzzy-matched embedding: '%s' -> '%s'", lookup, key)
                    return emb.to(device=self.device, dtype=target_dtype)
        logger.warning(
            "No embedding found for task='%s' prompt='%s', using zeros",
            task_name,
            prompt,
        )
        return torch.zeros((1, 512, self.cfg.text_emb_dim), device=self.device, dtype=target_dtype)

    @staticmethod
    def _to_uint8_image(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return image
        if image.max() <= 1.0:
            return (image * 255.0).clip(0, 255).astype(np.uint8)
        return image.clip(0, 255).astype(np.uint8)

    def _extract_obs_dict(self, payload_obs: dict[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray]:
        if "observation.images.cam_high" in payload_obs:
            cams = {
                "cam_high": self._to_uint8_image(np.asarray(payload_obs["observation.images.cam_high"])),
                "cam_left_wrist": self._to_uint8_image(np.asarray(payload_obs["observation.images.cam_left_wrist"])),
                "cam_right_wrist": self._to_uint8_image(np.asarray(payload_obs["observation.images.cam_right_wrist"])),
            }
            state = np.asarray(payload_obs["observation.state"], dtype=np.float32).reshape(-1)
            return cams, state

        if "image" in payload_obs and "state" in payload_obs:
            image_dict = payload_obs["image"]
            cams = {
                "cam_high": self._to_uint8_image(np.asarray(image_dict["base_0_rgb"])),
                "cam_left_wrist": self._to_uint8_image(np.asarray(image_dict["left_wrist_0_rgb"])),
                "cam_right_wrist": self._to_uint8_image(np.asarray(image_dict["right_wrist_0_rgb"])),
            }
            state = np.asarray(payload_obs["state"], dtype=np.float32).reshape(-1)
            return cams, state

        raise KeyError("Unsupported observation format.")

    def _append_single_observation(self, payload_obs: dict[str, Any]) -> None:
        cams, state = self._extract_obs_dict(payload_obs)
        self.raw_obs_history.append(cams)
        self.latest_state = state

    @torch.no_grad()
    def _encode_single_observation_to_latent(self, obs: dict[str, np.ndarray]) -> torch.Tensor:
        # Use only primary camera (cam_high); encoding pipeline keeps exactly the same as before.
        img = obs["cam_high"]
        img_t = torch.from_numpy(img).to(self.device).float() / 127.5 - 1.0
        img_t = img_t.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # [1,3,1,H,W]
        lat = self.vae.encode(img_t)[:, :, 0]  # [1,16,H',W']
        return lat[0].contiguous()  # [16,H',W']

    def _append_observation_payload(self, payload: Any) -> None:
        if payload is None:
            return
        if isinstance(payload, list):
            for one in payload:
                self._append_single_observation(one)
            return
        if isinstance(payload, dict):
            self._append_single_observation(payload)
            return
        raise TypeError(f"Unsupported obs payload type: {type(payload)}")

    def _get_last_k(self, values: list[Any], k: int) -> list[Any]:
        assert len(values) > 0
        out = values[-k:]
        while len(out) < k:
            out.append(out[-1])
        return out

    def _build_cond_latent(self) -> torch.Tensor:
        if len(self.latent_history) == 0:
            if len(self.raw_obs_history) == 0:
                raise RuntimeError("No observation available for latent encoding.")
            init_lat = self._encode_single_observation_to_latent(self.raw_obs_history[-1])
            self.latent_history.append(init_lat)
        frames = self._get_last_k(self.latent_history, self.cfg.num_cond_frames)
        cond_latent = torch.stack(frames, dim=1).unsqueeze(0).to(self.device)  # [1,16,4,H',W']
        return cond_latent

    def _build_model_batch(self, task_name: str | None, prompt: str | None) -> dict[str, torch.Tensor]:
        cond_latent = self._build_cond_latent()
        pred_placeholder = torch.zeros(
            (1, self.cfg.action_dim, self.cfg.num_pred_frames, cond_latent.shape[-2], cond_latent.shape[-1]),
            device=self.device,
            dtype=cond_latent.dtype,
        )
        latent_video = torch.cat([cond_latent, pred_placeholder], dim=2)

        if hasattr(self.model, "precision"):
            target_dtype = self.model.precision
        else:
            target_dtype = torch.bfloat16
        text_emb = self._get_text_embedding(task_name, prompt, target_dtype)

        batch = {
            "video": latent_video.to(dtype=target_dtype),
            "t5_text_embeddings": text_emb,
            "ai_caption": prompt or "",
            "fps": torch.tensor([16], device=self.device, dtype=target_dtype),
            "padding_mask": torch.zeros((1, 1, cond_latent.shape[-2], cond_latent.shape[-1]), device=self.device, dtype=target_dtype),
            "num_conditional_frames": self.cfg.num_cond_frames,
        }
        return batch

    def _predict_action(self, task_name: str | None, prompt: str | None) -> np.ndarray:
        batch = self._build_model_batch(task_name, prompt)
        with torch.inference_mode():
            latents = self.model.generate_samples_from_batch(
                batch,
                guidance=self.cfg.guidance,
                num_steps=self.cfg.num_steps,
                shift=self.cfg.shift,
                seed=self.cfg.seed + self.step_id,
            )

            pred_v = latents[:, :, self.cfg.num_cond_frames : self.cfg.num_cond_frames + self.cfg.num_pred_frames]
            prev_v = latents[:, :, self.cfg.num_cond_frames - 1 : self.cfg.num_cond_frames + self.cfg.num_pred_frames - 1]
            delta_v = rearrange(pred_v - prev_v, "b c t h w -> b t c h w").contiguous()

            if self.latest_state is None:
                raise RuntimeError("State is missing for action head inference.")
            state = self.latest_state.astype(np.float32)
            state = self.qnorm.norm_state(state)
            state_t = torch.from_numpy(state).to(self.device).unsqueeze(0)

            z_action = torch.zeros((1, 64, self.cfg.action_dim), device=self.device, dtype=torch.float32)
            timestep = torch.zeros((1,), device=self.device, dtype=torch.long)
            action = self.model.action_head(z_action, delta_v, state_t, timestep=timestep)[0]  # [64,16]

        action_np = action.detach().cpu().numpy().astype(np.float32)
        action_np = self.qnorm.unnorm_action(action_np)
        return action_np  # [64,16]

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("reset", False):
            self._reset()
            return {}

        if payload.get("compute_kv_cache", False):
            obs_seq = payload.get("obs", None)
            if isinstance(obs_seq, list):
                sampled = obs_seq[::2]  # 每2帧取1帧（时间降采样）
            elif obs_seq is None:
                sampled = []
            else:
                sampled = [obs_seq]
            for one in sampled:
                cams, _ = self._extract_obs_dict(one)
                self.raw_obs_history.append(cams)
                latent = self._encode_single_observation_to_latent(cams)
                self.latent_history.append(latent)
            return {}

        obs_payload = payload.get("obs", payload)
        self._append_observation_payload(obs_payload)
        if len(self.raw_obs_history) == 0:
            raise RuntimeError("No observation available for inference.")

        prompt = payload.get("prompt", None)
        if prompt is None and isinstance(obs_payload, dict):
            prompt = obs_payload.get("task", None)
        task_name = payload.get("task_name", None)
        if task_name is None and isinstance(obs_payload, dict):
            task_name = obs_payload.get("task_name", None)
        if task_name is None and isinstance(obs_payload, dict):
            task_name = obs_payload.get("task", None)
        action = self._predict_action(task_name, prompt)
        self.step_id += 1
        return {"action": action}


def parse_args() -> DeployConfig:
    parser = argparse.ArgumentParser("Cosmos RobotWin websocket server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--experiment_name", type=str, default="my_video_experiment")
    parser.add_argument("--config_file", type=str, default="configs/config.py")
    parser.add_argument("--video_ckpt", type=str, required=True)
    parser.add_argument("--action_head_ckpt", type=str, required=True)
    parser.add_argument("--vae_path", type=str, default="/home/jwhe/linyihan/cosmos/tokenizer.pth")
    parser.add_argument("--stats_json", type=str, default="")
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--text_emb_dim", type=int, default=100352)
    parser.add_argument("--text_emb_pt", type=str, default="")
    args = parser.parse_args()
    return DeployConfig(**vars(args))


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    cfg = parse_args()
    policy = CosmosRobotWinServer(cfg)
    server = WebsocketPolicyServer(policy, host=cfg.host, port=cfg.port, metadata={"name": "cosmos_robotwin_server"})
    logger.info("Starting websocket server on %s:%d", cfg.host, cfg.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
