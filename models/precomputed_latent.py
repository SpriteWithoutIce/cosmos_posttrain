from __future__ import annotations

import os
import random
import torch
import torch.nn.functional as F
from torch import Tensor

from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import Video2WorldCondition
from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import (
    NUM_CONDITIONAL_FRAMES_KEY,
    Video2WorldModelRectifiedFlow,
)
from models.action_head import ActionMIPHead


class IdentityLatentTokenizer(torch.nn.Module):
    """Identity encoder with optional real VAE decoder for visualization callbacks."""

    def __init__(
        self,
        latent_ch: int = 16,
        spatial_compression_factor: int = 8,
        name: str = "identity_latent_tokenizer",
        enable_decode: bool = False,
        vae_pth: str | None = None,
        temporal_window: int = 16,
    ):
        super().__init__()
        self._latent_ch = latent_ch
        self._spatial_compression_factor = spatial_compression_factor
        self.name = name
        self._decoder = None

        if enable_decode:
            from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface

            decoder_kwargs = {"temporal_window": temporal_window}
            if vae_pth:
                decoder_kwargs["vae_pth"] = vae_pth
            self._decoder = Wan2pt1VAEInterface(**decoder_kwargs)

    @property
    def latent_ch(self) -> int:
        return self._latent_ch

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self._decoder is None:
            raise RuntimeError(
                "IdentityLatentTokenizer.decode() requested but decoder is disabled. "
                "Set enable_decode=True and provide vae_pth."
            )
        return self._decoder.decode(latent)

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        if self._decoder is None:
            return num_pixel_frames
        return self._decoder.get_latent_num_frames(num_pixel_frames)

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        if self._decoder is None:
            return num_latent_frames
        return self._decoder.get_pixel_num_frames(num_latent_frames)

    @property
    def spatial_compression_factor(self) -> int:
        return self._spatial_compression_factor


class PrecomputedLatentVideo2WorldModel(Video2WorldModelRectifiedFlow):
    """Video2World model variant that consumes precomputed latents directly."""

    def __init__(
        self,
        *args,
        action_head_enabled: bool = False,
        action_head_cfg: dict | None = None,
        action_loss_weight: float = 1.0,
        action_head_save_every: int = 0,
        action_head_save_dir: str | None = None,
        action_head_load_path: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.action_head_enabled = bool(action_head_enabled)
        self.action_loss_weight = float(action_loss_weight)
        self.action_head_save_every = int(action_head_save_every)
        self.action_head_save_dir = action_head_save_dir
        self.action_head_load_path = action_head_load_path

        self.action_head = None
        self._action_head_debug = os.environ.get("ACTION_HEAD_DEBUG", "0") == "1"
        self._action_head_debug_printed = False
        self._action_head_log_every = int(os.environ.get("ACTION_HEAD_LOG_EVERY", "50"))
        self._action_head_wandb_log = os.environ.get("ACTION_HEAD_WANDB_LOG", "1") == "1"
        if self.action_head_enabled:
            cfg = dict(action_head_cfg or {})
            self.action_head = ActionMIPHead(**cfg)
            if self.action_head_load_path and os.path.isfile(self.action_head_load_path):
                self.load_action_head(self.action_head_load_path, strict=True)

    def _normalize_video_databatch_inplace(self, data_batch: dict[str, Tensor], input_key: str = None) -> None:
        # Sampling callbacks call this before generation. For latent-direct training,
        # the "video" tensor is already latent, not uint8 pixels.
        del input_key
        return

    @torch.no_grad()
    def generate_samples_from_batch(self, data_batch: dict, **kwargs) -> torch.Tensor:
        # Callback does not pass seed; use random seed each call so repeated guidance
        # entries can generate multiple diverse open-loop samples.
        if "seed" not in kwargs or kwargs["seed"] is None:
            kwargs["seed"] = random.randint(1, 2**31 - 1)
        return super().generate_samples_from_batch(data_batch, **kwargs)

    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> tuple[Tensor, Tensor, Video2WorldCondition]:
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key

        if input_key not in data_batch:
            raise KeyError(f"Missing required key '{input_key}' in batch")

        # Precomputed latents are already model-ready; skip normalization and VAE encoding.
        latent_state = data_batch[input_key].to(**self.tensor_kwargs).contiguous().float()
        raw_state = latent_state
        # EveryNDrawSample stacks generated sample with raw_data for visualization.
        # Decode raw latents only in no-grad context (sampling callbacks) to avoid training overhead.
        if not torch.is_grad_enabled():
            raw_state = self.decode(latent_state).contiguous().float()

        condition = self.conditioner(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        condition = condition.set_video_condition(
            gt_frames=latent_state.to(**self.tensor_kwargs),
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None),
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        return raw_state, latent_state, condition

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        if not isinstance(sd, dict):
            return sd
        return {k: v for k, v in sd.items() if not k.startswith("action_head.")}

    def load_state_dict(self, state_dict, strict: bool = True):
        if not self.action_head_enabled:
            return super().load_state_dict(state_dict, strict=strict)

        # Keep video checkpoint strictness while allowing action_head to be absent.
        result = super().load_state_dict(state_dict, strict=False)
        if strict:
            missing = [k for k in result.missing_keys if not k.startswith("action_head.")]
            unexpected = [k for k in result.unexpected_keys if not k.startswith("action_head.")]
            if missing or unexpected:
                raise RuntimeError(
                    f"Error(s) in loading state_dict for {self.__class__.__name__}: "
                    f"missing={missing}, unexpected={unexpected}"
                )
        return result

    @staticmethod
    def _is_rank0() -> bool:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0

    @staticmethod
    def _extract_velocity_tensor(output_batch: dict) -> Tensor | None:
        candidate_keys = (
            "vt_pred_B_C_T_H_W",
            "vt_pred",
            "velocity_pred",
            "pred_velocity",
            "pred_v",
            "net_output_B_C_T_H_W",
        )
        for key in candidate_keys:
            val = output_batch.get(key, None)
            if torch.is_tensor(val) and val.ndim == 5:
                return val
        return None

    def _compute_delta_v(self, output_batch: dict, num_cond: int, num_pred: int) -> Tensor:
        v_pred = self._extract_velocity_tensor(output_batch)
        if v_pred is None:
            if self._action_head_debug and self._is_rank0():
                print(
                    "[action-head][debug] output_batch keys:",
                    sorted(list(output_batch.keys())),
                    flush=True,
                )
            raise KeyError(
                "Cannot find predicted velocity tensor in output_batch. "
                "Expected one of: vt_pred_B_C_T_H_W / vt_pred / velocity_pred / pred_velocity / pred_v."
            )
        # v_pred: [B, C, T, H, W] -> [B, T, C] via global spatial pooling.
        v_seq = v_pred.mean(dim=(-1, -2)).transpose(1, 2).contiguous()
        t_total = v_seq.shape[1]
        if num_cond + num_pred > t_total:
            raise ValueError(f"Invalid cond/pred split: cond={num_cond}, pred={num_pred}, total={t_total}")

        pred_v = v_seq[:, num_cond : num_cond + num_pred, :]
        prev_v = v_seq[:, num_cond - 1 : num_cond + num_pred - 1, :]
        return pred_v - prev_v

    def _maybe_save_action_head(self, iteration: int) -> None:
        if not self.action_head_enabled or self.action_head is None:
            return
        if self.action_head_save_every <= 0 or iteration <= 0 or (iteration % self.action_head_save_every != 0):
            return
        if not self._is_rank0():
            return
        save_dir = self.action_head_save_dir
        if not save_dir:
            return
        os.makedirs(save_dir, exist_ok=True)
        fp = os.path.join(save_dir, f"action_head_iter_{iteration:07d}.pt")
        payload = {
            "iteration": int(iteration),
            "action_head": self.action_head.state_dict(),
        }
        torch.save(payload, fp)
        print(f"[action-head] saved: {fp}", flush=True)

    def load_action_head(self, checkpoint_path: str, strict: bool = True) -> None:
        if not self.action_head_enabled or self.action_head is None:
            return
        obj = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(obj, dict) and "action_head" in obj:
            sd = obj["action_head"]
        else:
            sd = obj
        self.action_head.load_state_dict(sd, strict=strict)
        print(f"[action-head] loaded: {checkpoint_path}", flush=True)

    def training_step(self, data_batch: dict, iteration: int = 0):
        output_batch, loss = super().training_step(data_batch, iteration)
        if not self.action_head_enabled or self.action_head is None:
            return output_batch, loss

        actions = data_batch.get("actions", None)
        states = data_batch.get("states", None)
        if actions is None or states is None:
            raise KeyError("Action head enabled, but batch is missing 'actions' or 'states'.")
        if actions.ndim != 3 or states.ndim != 2:
            raise ValueError(f"Invalid action/state shapes: actions={tuple(actions.shape)}, states={tuple(states.shape)}")

        actions = actions.to(loss.device).float()
        states = states.to(loss.device).float()

        num_cond = int(getattr(self.config, "min_num_conditional_frames", 4))
        num_pred = 8
        delta_v = self._compute_delta_v(output_batch, num_cond=num_cond, num_pred=num_pred).to(loss.device)
        if self._action_head_debug and (not self._action_head_debug_printed) and self._is_rank0():
            print(
                "[action-head][debug]",
                f"actions={tuple(actions.shape)} states={tuple(states.shape)} delta_v={tuple(delta_v.shape)}",
                flush=True,
            )
            self._action_head_debug_printed = True

        noise = torch.randn_like(actions)
        z1 = noise
        z2 = 0.9 * actions + noise

        pred1 = self.action_head(z1, delta_v, states)
        pred2 = self.action_head(z2, delta_v, states)

        action_loss = 0.5 * (F.mse_loss(pred1, actions) + F.mse_loss(pred2, actions))
        total_loss = loss + self.action_loss_weight * action_loss

        output_batch["action_loss"] = action_loss.detach()
        output_batch["video_loss"] = loss.detach()
        output_batch["total_loss"] = total_loss.detach()
        output_batch["metrics/action_loss"] = output_batch["action_loss"]
        output_batch["metrics/video_loss"] = output_batch["video_loss"]
        output_batch["metrics/total_loss"] = output_batch["total_loss"]

        if self._is_rank0() and self._action_head_log_every > 0 and iteration % self._action_head_log_every == 0:
            print(
                (
                    f"[action-head] iter={iteration} "
                    f"video_loss={float(loss.detach().item()):.6f} "
                    f"action_loss={float(action_loss.detach().item()):.6f} "
                    f"total_loss={float(total_loss.detach().item()):.6f}"
                ),
                flush=True,
            )
            if self._action_head_wandb_log:
                try:
                    import wandb  # type: ignore

                    if wandb.run is not None:
                        wandb.log(
                            {
                                "train/video_loss": float(loss.detach().item()),
                                "train/action_loss": float(action_loss.detach().item()),
                                "train/total_loss": float(total_loss.detach().item()),
                            },
                            step=int(iteration),
                        )
                except Exception:
                    pass

        self._maybe_save_action_head(iteration=iteration)
        return output_batch, total_loss
