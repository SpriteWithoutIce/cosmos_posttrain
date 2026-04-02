from __future__ import annotations

import os
import random

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.distributions import Beta

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.utils.optim_instantiate import get_base_scheduler
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import Video2WorldCondition
from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import (
    NUM_CONDITIONAL_FRAMES_KEY,
    Video2WorldModelRectifiedFlow,
)
from models.action_head import build_action_head


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
        action_head_type: str = "flow_matching",
        action_head_cfg: dict | None = None,
        action_head_lr: float = 1e-4,
        action_loss_weight: float = 1.0,
        action_head_timestep_mode: str = "uniform",
        action_head_fixed_timestep: float = 0.0,
        action_head_noise_beta_alpha: float = 1.5,
        action_head_noise_beta_beta: float = 1.0,
        action_head_noise_beta_s: float = 0.999,
        action_head_stop_gradient: bool = True,
        action_head_video_hidden_pred_only: bool = True,
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
        self.action_head_timestep_mode = str(action_head_timestep_mode)
        self.action_head_fixed_timestep = float(action_head_fixed_timestep)
        self.action_head_noise_beta_alpha = float(action_head_noise_beta_alpha)
        self.action_head_noise_beta_beta = float(action_head_noise_beta_beta)
        self.action_head_noise_beta_s = float(action_head_noise_beta_s)
        self.action_head_stop_gradient = bool(action_head_stop_gradient)
        self.action_head_video_hidden_pred_only = bool(action_head_video_hidden_pred_only)
        self._action_beta_dist = Beta(self.action_head_noise_beta_alpha, self.action_head_noise_beta_beta)
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
            cfg["n_blocks"] = len(self.net.blocks)
            cfg.setdefault("video_hidden_dim", getattr(self.net, "model_channels", cfg.get("video_hidden_dim", 2048)))
            self.action_head = build_action_head(self.action_head_type, **cfg)
            if self.action_head_load_path and os.path.isfile(self.action_head_load_path):
                self.load_action_head(self.action_head_load_path, strict=True)

    def init_optimizer_scheduler(
        self, optimizer_config: LazyDict, scheduler_config: LazyDict
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        if not self.action_head_enabled or self.action_head is None:
            return super().init_optimizer_scheduler(optimizer_config, scheduler_config)

        lr_video = float(optimizer_config.get("lr", 2e-5))
        weight_decay = float(optimizer_config.get("weight_decay", 0.0))
        optim_type = str(optimizer_config.get("optim_type", "adamw")).lower()

        video_params = [p for p in self.net.parameters() if p.requires_grad]
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

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        if not isinstance(sd, dict):
            return sd
        return {k: v for k, v in sd.items() if not k.startswith("action_head.")}

    def _is_ignorable_missing_key(self, key: str) -> bool:
        ignored_prefixes = (
            "action_head.",
            "net.video_action_conditioner.",
            "net_ema.video_action_conditioner.",
        )
        return key.startswith(ignored_prefixes)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False, **kwargs):
        result = super().load_state_dict(state_dict, strict=False, assign=assign, **kwargs)
        if strict:
            missing = [k for k in result.missing_keys if not self._is_ignorable_missing_key(k)]
            unexpected = [k for k in result.unexpected_keys if not self._is_ignorable_missing_key(k)]
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
        v = output_batch.get("model_pred", None)
        if torch.is_tensor(v) and v.ndim == 5:
            return v
        return None

    @staticmethod
    def _prepare_batch_actions(data_batch: dict, device: torch.device | None = None) -> Tensor | None:
        actions = data_batch.get("actions", None)
        if actions is None:
            return None
        if not torch.is_tensor(actions):
            raise TypeError(f"Expected 'actions' to be a tensor, got {type(actions)}")
        actions = actions.float()
        if device is not None:
            actions = actions.to(device=device)
        return actions

    def _maybe_apply_action_conditioning(self, net_kwargs: dict, action: Tensor | None) -> dict:
        if action is None:
            return net_kwargs
        if getattr(self.net, "supports_action_conditioning", False):
            net_kwargs = dict(net_kwargs)
            net_kwargs["action"] = action.to(**self.tensor_kwargs)
        return net_kwargs

    def _forward_net(
        self,
        xt_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition,
        action: Tensor | None = None,
        collect_last_hidden: bool = False,
    ) -> tuple[torch.Tensor, Tensor | list[Tensor] | None]:
        net_kwargs = condition.to_dict()
        if collect_last_hidden:
            if getattr(self.action_head, "requires_video_hidden_layers", False):
                net_kwargs["intermediate_feature_ids"] = list(range(len(self.net.blocks)))
            else:
                net_kwargs["intermediate_feature_ids"] = [len(self.net.blocks) - 1]
        net_kwargs = self._maybe_apply_action_conditioning(net_kwargs, action)

        net_out = self.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps_B_T,
            **net_kwargs,
        )

        if collect_last_hidden:
            net_output_B_C_T_H_W, hidden_list = net_out
            if getattr(self.action_head, "requires_video_hidden_layers", False):
                return net_output_B_C_T_H_W.float(), hidden_list
            last_hidden = hidden_list[-1] if hidden_list else None
            return net_output_B_C_T_H_W.float(), last_hidden
        return net_out.float(), None

    def denoise(
        self,
        noise: torch.Tensor,
        xt_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition,
        action: Tensor | None = None,
        collect_last_hidden: bool = False,
    ):
        if condition.is_video:
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(xt_B_C_T_H_W)
            if not condition.use_video_condition:
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, c_channels, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, c_channels, 1, 1, 1).type_as(
                xt_B_C_T_H_W
            )
            xt_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask + xt_B_C_T_H_W * (1 - condition_video_mask)

            if self.config.conditional_frame_timestep >= 0:
                condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
                timestep_cond_B_1_T_1_1 = (
                    torch.ones_like(condition_video_mask_B_1_T_1_1) * self.config.conditional_frame_timestep
                )
                timesteps_B_1_T_1_1 = timestep_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + timesteps_B_T * (
                    1 - condition_video_mask_B_1_T_1_1
                )
                timesteps_B_T = timesteps_B_1_T_1_1.squeeze()
                timesteps_B_T = timesteps_B_T.unsqueeze(0) if timesteps_B_T.ndim == 1 else timesteps_B_T

        net_output_B_C_T_H_W, last_hidden = self._forward_net(
            xt_B_C_T_H_W=xt_B_C_T_H_W,
            timesteps_B_T=timesteps_B_T,
            condition=condition,
            action=action,
            collect_last_hidden=collect_last_hidden,
        )

        if condition.is_video and self.config.denoise_replace_gt_frames:
            gt_frames_x0 = condition.gt_frames.type_as(net_output_B_C_T_H_W)
            gt_frames_velocity = noise - gt_frames_x0
            net_output_B_C_T_H_W = gt_frames_velocity * condition_video_mask + net_output_B_C_T_H_W * (1 - condition_video_mask)

        if collect_last_hidden:
            return net_output_B_C_T_H_W, last_hidden
        return net_output_B_C_T_H_W

    def get_velocity_fn_from_batch(
        self,
        data_batch: dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ):
        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
        else:
            num_conditional_frames = 1

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        action = self._prepare_batch_actions(data_batch, device=self.tensor_kwargs["device"])
        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        _, x0, _ = self.get_data_and_condition(data_batch)
        condition = condition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        uncondition = uncondition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        condition = condition.edit_for_inference(is_cfg_conditional=True, num_conditional_frames=num_conditional_frames)
        uncondition = uncondition.edit_for_inference(
            is_cfg_conditional=False, num_conditional_frames=num_conditional_frames
        )

        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)

        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            cond_v = self.denoise(noise, noise_x, timestep, condition, action=action)
            uncond_v = self.denoise(noise, noise_x, timestep, uncondition, action=action)
            velocity_pred = cond_v + guidance * (cond_v - uncond_v)
            return velocity_pred

        return velocity_fn

    def _sample_action_head_timestep(self, batch_size: int, device: torch.device) -> Tensor:
        mode = self.action_head_timestep_mode.lower()
        if mode == "fixed":
            t = float(self.action_head_fixed_timestep)
            buckets = float(getattr(self.action_head, "timestep_buckets", 1000) - 1)
            if t > 1.0 and buckets > 0:
                t = t / buckets
            return torch.full((batch_size,), max(0.0, min(1.0, t)), device=device, dtype=torch.float32)
        if mode in {"random", "uniform"}:
            return torch.rand(batch_size, device=device, dtype=torch.float32)
        if mode == "beta":
            return self._action_beta_dist.sample([batch_size]).to(device=device, dtype=torch.float32)
        raise ValueError(f"Unsupported action head timestep mode: {self.action_head_timestep_mode}")

    def _reshape_last_hidden(self, last_hidden: Tensor, xt_B_C_T_H_W: Tensor) -> Tensor:
        if last_hidden.ndim != 3:
            raise ValueError(f"Expected last_hidden [B,N,D], got {tuple(last_hidden.shape)}")
        patch_t = int(getattr(self.net, "patch_temporal", 1))
        patch_s = int(getattr(self.net, "patch_spatial", 1))
        t = xt_B_C_T_H_W.shape[2] // patch_t
        h = xt_B_C_T_H_W.shape[3] // patch_s
        w = xt_B_C_T_H_W.shape[4] // patch_s
        expected_tokens = t * h * w
        if last_hidden.shape[1] != expected_tokens:
            raise ValueError(
                f"Last hidden token count mismatch: got {last_hidden.shape[1]}, expected {expected_tokens} "
                f"for latent grid {(t, h, w)}"
            )
        return last_hidden.view(last_hidden.shape[0], t, h, w, last_hidden.shape[-1])

    def _extract_action_head_video_tokens(
        self,
        last_hidden: Tensor,
        xt_B_C_T_H_W: Tensor,
        num_cond: int,
        num_pred: int,
    ) -> Tensor:
        hidden_grid = self._reshape_last_hidden(last_hidden, xt_B_C_T_H_W)
        if self.action_head_video_hidden_pred_only:
            hidden_grid = hidden_grid[:, num_cond : num_cond + num_pred]
        return hidden_grid.contiguous()

    def _extract_action_head_video_hidden_layers(
        self,
        hidden_layers: list[Tensor] | tuple[Tensor, ...],
        xt_B_C_T_H_W: Tensor,
        num_cond: int,
    ) -> list[Tensor]:
        if not isinstance(hidden_layers, (list, tuple)) or len(hidden_layers) == 0:
            raise ValueError("Expected non-empty list of video hidden layers.")
        extracted_layers: list[Tensor] = []
        for layer_hidden in hidden_layers:
            hidden_grid = self._reshape_last_hidden(layer_hidden, xt_B_C_T_H_W)
            # Only use conditional frames (clean, no noise) to avoid train/test mismatch
            extracted_layers.append(hidden_grid[:, :num_cond].contiguous())
        return extracted_layers

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

    def _forward_video_training(
        self,
        data_batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, Tensor | None, Tensor]:
        if self.text_encoder is not None and self.config.text_encoder_config.compute_online:
            text_embeddings = self.text_encoder.compute_text_embeddings_online(data_batch, self.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")

        _, x0_B_C_T_H_W, condition = self.get_data_and_condition(data_batch)
        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), **self.tensor_kwargs_fp32)
        batch_size = x0_B_C_T_H_W.size()[0]
        t_B = self.rectified_flow.sample_train_time(batch_size).to(**self.tensor_kwargs_fp32)
        t_B = rearrange(t_B, "b -> b 1")

        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B = self.broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B
        )
        timesteps = self.rectified_flow.get_discrete_timestamp(t_B, self.tensor_kwargs_fp32)

        if self.config.use_high_sigma_strategy:
            raise NotImplementedError("High sigma strategy is buggy when using CP")

        sigmas = self.rectified_flow.get_sigmas(timesteps, self.tensor_kwargs_fp32)
        timesteps = rearrange(timesteps, "b -> b 1")
        sigmas = rearrange(sigmas, "b -> b 1")
        xt_B_C_T_H_W, vt_B_C_T_H_W = self.rectified_flow.get_interpolation(epsilon_B_C_T_H_W, x0_B_C_T_H_W, sigmas)

        actions = self._prepare_batch_actions(data_batch, device=xt_B_C_T_H_W.device)
        vt_pred_B_C_T_H_W, last_hidden = self.denoise(
            noise=epsilon_B_C_T_H_W,
            xt_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps,
            condition=condition,
            action=actions,
            collect_last_hidden=self.action_head_enabled and self.action_head is not None,
        )

        time_weights_B = self.rectified_flow.train_time_weight(timesteps, self.tensor_kwargs_fp32)
        per_instance_loss = torch.mean(
            (vt_pred_B_C_T_H_W - vt_B_C_T_H_W) ** 2, dim=list(range(1, vt_pred_B_C_T_H_W.dim()))
        )
        loss = torch.mean(time_weights_B * per_instance_loss)
        output_batch = {
            "x0": x0_B_C_T_H_W,
            "xt": xt_B_C_T_H_W,
            "sigma": sigmas,
            "condition": condition,
            "model_pred": vt_pred_B_C_T_H_W,
            "edm_loss": loss,
            "timesteps": timesteps,
            "per_instance_loss": per_instance_loss,
            "n_cond_frames": condition.num_conditional_frames_B,
        }
        return output_batch, loss, last_hidden, xt_B_C_T_H_W

    def training_step(self, data_batch: dict, iteration: int = 0):
        """
        Joint training: Video and Action are generated simultaneously.
        - Video uses GT action as condition (via timestep embedding)
        - Action uses Video conditional frames as condition
        - Both losses are optimized together
        """
        if not self.action_head_enabled or self.action_head is None:
            # Fallback to original video-only training
            output_batch, loss, _, _ = self._forward_video_training(data_batch)
            return output_batch, loss

        # 1. Get GT video and GT action
        actions_gt = data_batch.get("actions", None)
        if actions_gt is None:
            raise KeyError("Action head enabled, but batch is missing 'actions'.")
        if not torch.is_tensor(actions_gt) or actions_gt.ndim != 3:
            raise ValueError(f"Invalid action shape: {tuple(actions_gt.shape) if torch.is_tensor(actions_gt) else None}")
        
        # 2. Prepare video data
        if self.text_encoder is not None and self.config.text_encoder_config.compute_online:
            text_embeddings = self.text_encoder.compute_text_embeddings_online(data_batch, self.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")

        _, x0_video, condition = self.get_data_and_condition(data_batch)
        epsilon_video = torch.randn(x0_video.size(), **self.tensor_kwargs_fp32)
        batch_size = x0_video.size()[0]
        
        # 3. Sample timesteps for video and action (independent)
        t_video = self.rectified_flow.sample_train_time(batch_size).to(**self.tensor_kwargs_fp32)
        t_video = rearrange(t_video, "b -> b 1")
        t_action = self._sample_action_head_timestep(batch_size=batch_size, device=x0_video.device)
        
        # 4. Broadcast for model parallelism
        x0_video, condition, epsilon_video, t_video = self.broadcast_split_for_model_parallelsim(
            x0_video, condition, epsilon_video, t_video
        )
        timesteps_video = self.rectified_flow.get_discrete_timestamp(t_video, self.tensor_kwargs_fp32)
        
        # 5. Prepare noisy video and action
        sigmas_video = self.rectified_flow.get_sigmas(timesteps_video, self.tensor_kwargs_fp32)
        timesteps_video = rearrange(timesteps_video, "b -> b 1")
        sigmas_video = rearrange(sigmas_video, "b -> b 1")
        xt_video, vt_video = self.rectified_flow.get_interpolation(epsilon_video, x0_video, sigmas_video)
        
        # Action: flow matching setup
        actions_gt = actions_gt.to(xt_video.device).float()
        noise_action = torch.randn_like(actions_gt)
        interp_action = t_action.view(-1, 1, 1)
        xt_action = (1.0 - interp_action) * noise_action + interp_action * actions_gt
        target_velocity_action = actions_gt - noise_action
        
        # 6. Video forward with GT action as condition
        actions_gt_for_video = actions_gt if not self.action_head_stop_gradient else actions_gt.detach()
        vt_pred_video, video_hidden = self.denoise(
            noise=epsilon_video,
            xt_B_C_T_H_W=xt_video.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps_video,
            condition=condition,
            action=actions_gt_for_video,  # GT action conditions video
            collect_last_hidden=True,
        )
        
        # 7. Compute video loss
        time_weights_video = self.rectified_flow.train_time_weight(timesteps_video, self.tensor_kwargs_fp32)
        per_instance_video_loss = torch.mean(
            (vt_pred_video - vt_video) ** 2, dim=list(range(1, vt_pred_video.dim()))
        )
        loss_video = torch.mean(time_weights_video * per_instance_video_loss)
        
        # 8. Action forward with video conditional frames
        num_cond = int(getattr(self.config, "min_num_conditional_frames", 4))
        if getattr(self.action_head, "requires_video_hidden_layers", False):
            video_tokens = self._extract_action_head_video_hidden_layers(video_hidden, xt_video, num_cond=num_cond)
        else:
            num_pred = max(1, actions_gt.shape[1] // max(1, int(getattr(self.action_head, "actions_per_latent", 8))))
            video_tokens = self._extract_action_head_video_tokens(video_hidden, xt_video, num_cond=num_cond, num_pred=num_pred)
        
        # 9. Action head forward
        pred_velocity_action = self.action_head(
            xt_action,
            video_tokens,
            state_vec=None,
            timestep=t_action,
        )
        
        # 10. Compute action loss
        loss_action = F.mse_loss(pred_velocity_action, target_velocity_action)
        
        # 11. Total loss (joint optimization)
        total_loss = loss_video + self.action_loss_weight * loss_action
        
        # 12. Prepare output
        output_batch = {
            "x0": x0_video,
            "xt": xt_video,
            "sigma": sigmas_video,
            "condition": condition,
            "model_pred": vt_pred_video,
            "video_loss": loss_video.detach(),
            "action_loss": loss_action.detach(),
            "total_loss": total_loss.detach(),
            "timesteps": timesteps_video,
            "per_instance_loss": per_instance_video_loss,
            "n_cond_frames": condition.num_conditional_frames_B,
            "metrics/video_loss": loss_video.detach(),
            "metrics/action_loss": loss_action.detach(),
            "metrics/total_loss": total_loss.detach(),
            "metrics/action_timestep_mean": t_action.detach().mean(),
        }
        
        # 13. Logging
        if self._is_rank0() and self._action_head_log_every > 0 and iteration % self._action_head_log_every == 0:
            print(
                f"[joint] iter={iteration} "
                f"video_loss={float(loss_video.detach().item()):.6f} "
                f"action_loss={float(loss_action.detach().item()):.6f} "
                f"total_loss={float(total_loss.detach().item()):.6f}",
                flush=True,
            )
            if self._action_head_wandb_log:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log(
                            {
                                "train/video_loss": float(loss_video.detach().item()),
                                "train/action_loss": float(loss_action.detach().item()),
                                "train/action_timestep_mean": float(t_action.detach().mean().item()),
                                "train/total_loss": float(total_loss.detach().item()),
                            },
                            step=int(iteration),
                        )
                except Exception:
                    pass
        
        self._maybe_save_action_head(iteration=iteration)
        return output_batch, total_loss
