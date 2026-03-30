from __future__ import annotations

import os
import random
import torch
import torch.nn.functional as F
from torch.distributions import Beta
from torch import Tensor
from einops import rearrange

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.utils.optim_instantiate import get_base_scheduler
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import Video2WorldCondition
from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import (
    NUM_CONDITIONAL_FRAMES_KEY,
    Video2WorldModelRectifiedFlow,
)

# ------------------------------------------------------------------
# Factories
# ------------------------------------------------------------------
from models.action_heads import build_action_head
from models.video_strategies import build_video_strategy
from models.video_strategies.base import BaseVideoStrategy
from models.action_heads.base import BaseActionHead


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
    """Video2World model variant that consumes precomputed latents directly.

    Supports pluggable action heads and video training strategies via registries.
    """

    def __init__(
        self,
        *args,
        # --- action head (factory-based) ---
        action_head_enabled: bool = False,
        action_head_type: str = "mip",
        action_head_cfg: dict | None = None,
        action_head_lr: float = 1e-4,
        action_loss_weight: float = 1.0,
        # --- video strategy (factory-based) ---
        video_strategy_type: str = "standard_rf",
        video_strategy_cfg: dict | None = None,
        # --- legacy params (forwarded to MIP head for compat) ---
        action_delta_video_t: float = 0.5,
        action_head_timestep_mode: str = "beta",
        action_head_fixed_timestep: int = 0,
        action_head_mip_gt_mix: float = 0.9,
        action_head_noise_beta_alpha: float = 1.5,
        action_head_noise_beta_beta: float = 1.0,
        action_head_noise_s: float = 0.999,
        # --- checkpoint ---
        action_head_save_every: int = 0,
        action_head_save_dir: str | None = None,
        action_head_load_path: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.action_head_enabled = bool(action_head_enabled)
        self.action_head_type = str(action_head_type)
        self.action_head_lr = float(action_head_lr)
        self.action_loss_weight = float(action_loss_weight)
        self.action_delta_video_t = float(action_delta_video_t)
        self.action_head_save_every = int(action_head_save_every)
        self.action_head_save_dir = action_head_save_dir
        self.action_head_load_path = action_head_load_path

        self._action_head_debug = os.environ.get("ACTION_HEAD_DEBUG", "0") == "1"
        self._action_head_debug_printed = False
        self._action_head_log_every = int(os.environ.get("ACTION_HEAD_LOG_EVERY", "50"))
        self._action_head_wandb_log = os.environ.get("ACTION_HEAD_WANDB_LOG", "1") == "1"

        # --- Build action head via factory ---
        self.action_head: BaseActionHead | None = None
        if self.action_head_enabled:
            head_cfg = dict(action_head_cfg or {})
            # Inject legacy MIP params into config if using MIP head
            if self.action_head_type == "mip":
                head_cfg.setdefault("mip_gt_mix", action_head_mip_gt_mix)
                head_cfg.setdefault("timestep_mode", action_head_timestep_mode)
                head_cfg.setdefault("fixed_timestep", action_head_fixed_timestep)
                head_cfg.setdefault("noise_beta_alpha", action_head_noise_beta_alpha)
                head_cfg.setdefault("noise_beta_beta", action_head_noise_beta_beta)
                head_cfg.setdefault("noise_s", action_head_noise_s)
            self.action_head = build_action_head(self.action_head_type, **head_cfg)

            if self.action_head_load_path and os.path.isfile(self.action_head_load_path):
                self.load_action_head(self.action_head_load_path, strict=True)

        # --- Build video strategy via factory ---
        self.video_strategy: BaseVideoStrategy | None = None
        if self.action_head_enabled:
            strategy_cfg = dict(video_strategy_cfg or {})
            # Inject delta_video_t for standard_rf compat
            vs_type = str(video_strategy_type)
            if vs_type == "standard_rf":
                strategy_cfg.setdefault("delta_video_t", action_delta_video_t)
            self.video_strategy = build_video_strategy(vs_type, **strategy_cfg)

            # If strategy owns extra modules (e.g. ActionTimestepInjector),
            # register them so they get saved / moved to device properly.
            if hasattr(self.video_strategy, "injector"):
                self.strategy_injector = self.video_strategy.injector

    # ------------------------------------------------------------------
    # Optimizer: grouped LR for video + action head + strategy modules
    # ------------------------------------------------------------------
    def init_optimizer_scheduler(
        self, optimizer_config: LazyDict, scheduler_config: LazyDict
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        if not self.action_head_enabled or self.action_head is None:
            return super().init_optimizer_scheduler(optimizer_config, scheduler_config)

        lr_video = float(optimizer_config.get("lr", 2e-5))
        weight_decay = float(optimizer_config.get("weight_decay", 0.0))
        optim_type = str(optimizer_config.get("optim_type", "adamw")).lower()

        video_params = [p for p in self.net.parameters() if p.requires_grad]
        # Include strategy injector params in video param group
        if self.video_strategy is not None and hasattr(self.video_strategy, "get_extra_parameters"):
            video_params.extend(self.video_strategy.get_extra_parameters())

        action_params = [p for p in self.action_head.parameters() if p.requires_grad]
        if len(action_params) == 0:
            return super().init_optimizer_scheduler(optimizer_config, scheduler_config)

        param_groups = [
            {"params": video_params, "lr": lr_video, "weight_decay": weight_decay},
            {"params": action_params, "lr": self.action_head_lr, "weight_decay": weight_decay},
        ]

        betas = tuple(optimizer_config.get("betas", [0.9, 0.99]))
        eps = float(optimizer_config.get("eps", 1e-8))
        if optim_type == "fusedadam":
            from cosmos_predict2._src.predict2.utils.fused_adam_dtensor import FusedAdam

            optimizer = FusedAdam(
                param_groups,
                betas=betas,
                eps=eps,
                master_weights=bool(optimizer_config.get("master_weights", True)),
                capturable=bool(optimizer_config.get("capturable", True)),
            )
        elif optim_type == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups,
                betas=betas,
                eps=eps,
                fused=bool(optimizer_config.get("fused", True)),
            )
        else:
            raise ValueError(f"Unsupported optim_type for grouped lr: {optim_type}")

        scheduler = get_base_scheduler(optimizer, self, scheduler_config)
        return optimizer, scheduler

    # ------------------------------------------------------------------
    # Data / normalization overrides
    # ------------------------------------------------------------------
    def _normalize_video_databatch_inplace(self, data_batch: dict[str, Tensor], input_key: str = None) -> None:
        del input_key
        return

    @torch.no_grad()
    def generate_samples_from_batch(self, data_batch: dict, **kwargs) -> torch.Tensor:
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

        latent_state = data_batch[input_key].to(**self.tensor_kwargs).contiguous().float()
        raw_state = latent_state
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

    # ------------------------------------------------------------------
    # Checkpoint: exclude action_head from video state_dict
    # ------------------------------------------------------------------
    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        if not isinstance(sd, dict):
            return sd
        return {k: v for k, v in sd.items() if not k.startswith("action_head.")}

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False, **kwargs):
        if not self.action_head_enabled:
            return super().load_state_dict(state_dict, strict=strict, assign=assign, **kwargs)

        result = super().load_state_dict(state_dict, strict=False, assign=assign, **kwargs)
        if strict:
            missing = [k for k in result.missing_keys if not k.startswith("action_head.") and not k.startswith("strategy_injector.")]
            unexpected = [k for k in result.unexpected_keys if not k.startswith("action_head.") and not k.startswith("strategy_injector.")]
            if missing or unexpected:
                raise RuntimeError(
                    f"Error(s) in loading state_dict for {self.__class__.__name__}: "
                    f"missing={missing}, unexpected={unexpected}"
                )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_rank0() -> bool:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0

    @staticmethod
    def _extract_velocity_tensor(output_batch: dict) -> Tensor | None:
        v = output_batch.get("model_pred", None)
        if torch.is_tensor(v) and v.ndim == 5:
            return v
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
                "Cannot find predicted velocity tensor in output_batch['model_pred']."
            )
        t_total = v_pred.shape[2]
        if num_cond + num_pred > t_total:
            raise ValueError(f"Invalid cond/pred split: cond={num_cond}, pred={num_pred}, total={t_total}")
        pred_v = v_pred[:, :, num_cond : num_cond + num_pred, :, :]
        prev_v = v_pred[:, :, num_cond - 1 : num_cond + num_pred - 1, :, :]
        delta_v = pred_v - prev_v
        return rearrange(delta_v, "b c t h w -> b t c h w").contiguous()

    def _compute_delta_v_at_fixed_video_t(self, output_batch: dict, num_cond: int, num_pred: int) -> Tensor:
        if "x0" not in output_batch or "condition" not in output_batch:
            raise KeyError("output_batch missing 'x0' or 'condition' for fixed-t action delta_v computation.")

        x0 = output_batch["x0"]
        condition = output_batch["condition"]
        batch_size = x0.shape[0]

        t_fix = max(0.0, min(1.0, self.action_delta_video_t))
        t_B = torch.full((batch_size, 1), t_fix, device=x0.device, dtype=torch.float32)
        timesteps = self.rectified_flow.get_discrete_timestamp(t_B, self.tensor_kwargs_fp32)
        sigmas = self.rectified_flow.get_sigmas(timesteps, self.tensor_kwargs_fp32)
        timesteps = timesteps.view(batch_size, 1)
        sigmas = sigmas.view(batch_size, 1)

        epsilon = torch.randn_like(x0, dtype=torch.float32)
        xt_fix, _ = self.rectified_flow.get_interpolation(epsilon, x0.to(dtype=torch.float32), sigmas)
        v_pred_fix = self.denoise(
            noise=epsilon,
            xt_B_C_T_H_W=xt_fix.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps,
            condition=condition,
        )

        pred_v = v_pred_fix[:, :, num_cond : num_cond + num_pred, :, :]
        prev_v = v_pred_fix[:, :, num_cond - 1 : num_cond + num_pred - 1, :, :]
        delta_v = pred_v - prev_v
        return rearrange(delta_v, "b c t h w -> b t c h w").contiguous()

    # ------------------------------------------------------------------
    # Action head checkpoint management
    # ------------------------------------------------------------------
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
            "action_head_type": self.action_head_type,
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

    # ------------------------------------------------------------------
    # Training step (strategy-agnostic)
    # ------------------------------------------------------------------
    def training_step(self, data_batch: dict, iteration: int = 0):
        # --- Prepare strategy (may inject action conditioning) ---
        extra = {}
        if self.action_head_enabled and self.video_strategy is not None:
            extra = self.video_strategy.prepare_video_forward(self, data_batch, {}, iteration)

        # If using action_conditioned_rf, inject the action bias into the
        # video model's timestep embedding via a hook.
        action_timestep_bias = extra.get("action_timestep_bias", None)
        hook_handle = None
        if action_timestep_bias is not None:
            def _inject_action_bias(module, args):
                # t_embedder returns (emb, adaln_lora) or just emb
                pass

            def _inject_bias_post(module, args, output):
                # output is tuple (emb_B_T_D, adaln_lora_B_T_3D) from t_embedder
                if isinstance(output, tuple):
                    emb, rest = output[0], output[1:]
                    bias = action_timestep_bias.unsqueeze(1).to(dtype=emb.dtype, device=emb.device)
                    emb = emb + bias
                    return (emb, *rest)
                else:
                    bias = action_timestep_bias.unsqueeze(1).to(dtype=output.dtype, device=output.device)
                    return output + bias

            hook_handle = self.net.t_embedder.register_forward_hook(_inject_bias_post)

        # --- Video forward ---
        output_batch, loss = super().training_step(data_batch, iteration)

        # Remove timestep injection hook
        if hook_handle is not None:
            hook_handle.remove()

        if not self.action_head_enabled or self.action_head is None:
            return output_batch, loss

        # --- Action head training ---
        actions = data_batch.get("actions", None)
        states = data_batch.get("states", None)
        if actions is None:
            raise KeyError("Action head enabled, but batch is missing 'actions'.")

        actions = actions.to(loss.device).float()
        if states is not None:
            states = states.to(loss.device).float()

        # Extract condition features via strategy
        condition_features = self.video_strategy.extract_action_condition(self, output_batch, extra)
        condition_features = condition_features.to(loss.device)

        # Detach if strategy says so (e.g. action_conditioned_rf)
        if self.video_strategy.should_detach_action_grad():
            condition_features = condition_features.detach()

        if self._action_head_debug and (not self._action_head_debug_printed) and self._is_rank0():
            print(
                "[action-head][debug]",
                f"type={self.action_head_type}",
                f"actions={tuple(actions.shape)}",
                f"cond={tuple(condition_features.shape)}",
                f"states={tuple(states.shape) if states is not None else None}",
                flush=True,
            )
            self._action_head_debug_printed = True

        # Compute action loss via the head's own compute_loss
        loss_dict = self.action_head.compute_loss(
            actions_gt=actions,
            condition_features=condition_features,
            state_vec=states,
        )
        action_loss = loss_dict["loss"]
        total_loss = loss + self.action_loss_weight * action_loss

        # Logging
        output_batch["video_loss"] = loss.detach()
        output_batch["action_loss"] = action_loss.detach()
        output_batch["total_loss"] = total_loss.detach()
        output_batch["metrics/video_loss"] = output_batch["video_loss"]
        output_batch["metrics/action_loss"] = output_batch["action_loss"]
        output_batch["metrics/total_loss"] = output_batch["total_loss"]
        for k, v in loss_dict.items():
            if k != "loss":
                output_batch[f"metrics/{k}"] = v

        if self._is_rank0() and self._action_head_log_every > 0 and iteration % self._action_head_log_every == 0:
            extra_info = " ".join(f"{k}={float(v):.6f}" for k, v in loss_dict.items() if k != "loss")
            print(
                f"[action-head] iter={iteration} "
                f"type={self.action_head_type} "
                f"video_loss={float(loss.detach().item()):.6f} "
                f"action_loss={float(action_loss.detach().item()):.6f} "
                f"{extra_info} "
                f"total_loss={float(total_loss.detach().item()):.6f}",
                flush=True,
            )
            if self._action_head_wandb_log:
                try:
                    import wandb

                    if wandb.run is not None:
                        log_dict = {
                            "train/video_loss": float(loss.detach().item()),
                            "train/action_loss": float(action_loss.detach().item()),
                            "train/total_loss": float(total_loss.detach().item()),
                        }
                        for k, v in loss_dict.items():
                            if k != "loss":
                                log_dict[f"train/{k}"] = float(v)
                        wandb.log(log_dict, step=int(iteration))
                except Exception:
                    pass

        self._maybe_save_action_head(iteration=iteration)
        return output_batch, total_loss
