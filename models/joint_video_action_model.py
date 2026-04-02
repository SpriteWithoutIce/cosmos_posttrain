# Joint Video-Action Generation Model with Cross-Attention
# Based on FastWAM-style MoT but adapted for Cosmos architecture

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Dict, List, Optional, Tuple

from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import (
    Video2WorldModelRectifiedFlow,
    Video2WorldModelRectifiedFlowConfig,
)
from models.action_head import build_action_head


class JointVideoActionModel(Video2WorldModelRectifiedFlow):
    """
    Joint video-action generation model.
    
    Key features:
    1. Video uses action as cross-attention condition (like FastWAM)
    2. Action uses video conditional frames as condition
    3. Joint training and inference
    """
    
    def __init__(
        self,
        *args,
        action_head_enabled: bool = True,
        action_head_type: str = "flow_matching",
        action_head_cfg: dict | None = None,
        action_head_lr: float = 1e-4,
        action_loss_weight: float = 1.0,
        action_head_timestep_mode: str = "uniform",
        action_head_fixed_timestep: float = 0.0,
        action_head_noise_beta_alpha: float = 1.5,
        action_head_noise_beta_beta: float = 1.0,
        action_head_noise_beta_s: float = 0.999,
        action_head_stop_gradient: bool = False,  # Must be False for joint training
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
        
        # Action embedding for video cross-attention
        if self.action_head_enabled:
            action_dim = action_head_cfg.get("action_dim", 16) if action_head_cfg else 16
            num_actions = action_head_cfg.get("num_actions", 64) if action_head_cfg else 64
            crossattn_dim = self.config.net.config.crossattn_emb_channels
            
            # Project action sequence to cross-attention dimension
            self.action_proj_for_video = nn.Sequential(
                nn.Linear(action_dim * num_actions, crossattn_dim),
                nn.LayerNorm(crossattn_dim),
                nn.GELU(),
                nn.Linear(crossattn_dim, crossattn_dim),
            )
            
            # Build action head
            cfg = dict(action_head_cfg or {})
            cfg["n_blocks"] = len(self.net.blocks)
            cfg.setdefault("video_hidden_dim", getattr(self.net, "model_channels", cfg.get("video_hidden_dim", 2048)))
            self.action_head = build_action_head(self.action_head_type, **cfg)
        
        self._action_head_log_every = 1
        self._action_head_wandb_log = True
    
    def _prepare_action_for_video_crossattn(self, action: torch.Tensor) -> torch.Tensor:
        """
        Prepare action embeddings for video cross-attention.
        
        Args:
            action: [B, num_actions, action_dim]
        
        Returns:
            action_emb: [B, 1, crossattn_dim] - single token representing action sequence
        """
        B = action.shape[0]
        action_flat = rearrange(action, "b n d -> b (n d)")
        action_emb = self.action_proj_for_video(action_flat)
        return action_emb.unsqueeze(1)  # [B, 1, D]
    
    def denoise(
        self,
        noise: torch.Tensor,
        xt_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition,
        action: torch.Tensor | None = None,  # For video cross-attention
        crossattn_emb: torch.Tensor | None = None,  # Text embeddings
        collect_last_hidden: bool = False,
    ):
        """
        Denoise with optional action conditioning via cross-attention.
        """
        # Handle conditional frames for video
        if condition.is_video:
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(xt_B_C_T_H_W)
            if not condition.use_video_condition:
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, c_channels, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(
                1, c_channels, 1, 1, 1
            ).type_as(xt_B_C_T_H_W)
            xt_B_C_T_H_W = (
                condition_state_in_B_C_T_H_W * condition_video_mask
                + xt_B_C_T_H_W * (1 - condition_video_mask)
            )

        # Prepare cross-attention embeddings: [Text, Action]
        context_input = crossattn_emb if crossattn_emb is not None else condition.to_dict().get("crossattn_emb")
        
        if action is not None and self.action_head_enabled:
            # Project action and concatenate to text embeddings
            action_emb = self._prepare_action_for_video_crossattn(action)
            if context_input is not None:
                context_input = torch.cat([context_input, action_emb], dim=1)
            else:
                context_input = action_emb
        
        # Forward through network
        net_kwargs = {
            "crossattn_emb": context_input,
            "fps": condition.to_dict().get("fps"),
            "padding_mask": condition.to_dict().get("padding_mask"),
        }
        
        if collect_last_hidden:
            net_kwargs["intermediate_feature_ids"] = list(range(len(self.net.blocks)))
        
        net_out = self.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps_B_T,
            **net_kwargs,
        )
        
        if collect_last_hidden:
            net_output, hidden_list = net_out
        else:
            net_output = net_out
            hidden_list = None
        
        # Replace conditional frames
        if condition.is_video and self.config.denoise_replace_gt_frames:
            gt_frames_x0 = condition.gt_frames.type_as(net_output)
            gt_frames_velocity = noise - gt_frames_x0
            net_output = (
                gt_frames_velocity * condition_video_mask
                + net_output * (1 - condition_video_mask)
            )
        
        if collect_last_hidden:
            return net_output.float(), hidden_list
        return net_output.float()
    
    def _extract_conditional_video_features(
        self,
        hidden_layers: List[torch.Tensor],
        xt_B_C_T_H_W: torch.Tensor,
        num_cond_frames: int = 4,
    ) -> List[torch.Tensor]:
        """Extract features from conditional frames only."""
        extracted = []
        for hidden in hidden_layers:
            # hidden: [B, T*H*W, D]
            B, N, D = hidden.shape
            T = xt_B_C_T_H_W.shape[2]
            H = xt_B_C_T_H_W.shape[3] // self.net.patch_spatial
            W = xt_B_C_T_H_W.shape[4] // self.net.patch_spatial
            
            # Reshape to [B, T, H*W, D]
            hidden = hidden.view(B, T, H * W, D)
            
            # Take only conditional frames
            hidden_cond = hidden[:, :num_cond_frames]  # [B, num_cond, H*W, D]
            
            # Reshape back to [B, num_cond*H*W, D]
            hidden_cond = hidden_cond.reshape(B, num_cond_frames * H * W, D)
            extracted.append(hidden_cond)
        
        return extracted
    
    def training_step(self, data_batch: dict, iteration: int = 0):
        """
        Joint training: Video and Action denoise simultaneously.
        """
        # 1. Get data
        x0_video = data_batch[self.input_data_key].to(**self.tensor_kwargs)
        actions_gt = data_batch.get("actions")
        
        if actions_gt is None or not self.action_head_enabled:
            # Fallback to video-only training
            return super().training_step(data_batch, iteration)
        
        actions_gt = actions_gt.to(x0_video.device).float()
        crossattn_emb = data_batch.get("t5_text_embeddings")
        
        B = x0_video.shape[0]
        
        # 2. Sample timesteps
        t_video = self.rectified_flow.sample_train_time(B).to(**self.tensor_kwargs_fp32)
        t_video = rearrange(t_video, "b -> b 1")
        t_action = torch.rand(B, device=x0_video.device, dtype=torch.float32)
        
        # 3. Prepare noisy inputs
        noise_video = torch.randn_like(x0_video)
        noise_action = torch.randn_like(actions_gt)
        
        sigmas_video = self.rectified_flow.get_sigmas(t_video, self.tensor_kwargs_fp32)
        xt_video, target_v_video = self.rectified_flow.get_interpolation(
            noise_video, x0_video, sigmas_video
        )
        
        xt_action = t_action.view(-1, 1, 1) * actions_gt + (1 - t_action.view(-1, 1, 1)) * noise_action
        target_v_action = actions_gt - noise_action
        
        # 4. Get condition for video
        _, _, condition = self.get_data_and_condition(data_batch)
        
        # 5. Video forward (conditioned on GT action via cross-attention)
        actions_for_video = actions_gt if not self.action_head_stop_gradient else actions_gt.detach()
        
        v_pred_video, hidden_layers = self.denoise(
            noise=noise_video,
            xt_B_C_T_H_W=xt_video,
            timesteps_B_T=t_video,
            condition=condition,
            action=actions_for_video,
            crossattn_emb=crossattn_emb,
            collect_last_hidden=True,
        )
        
        # Video loss
        loss_video = F.mse_loss(v_pred_video, target_v_video)
        
        # 6. Action forward (conditioned on video conditional frames)
        num_cond = self.config.min_num_conditional_frames
        video_features = self._extract_conditional_video_features(hidden_layers, xt_video, num_cond)
        
        v_pred_action = self.action_head(
            xt_action,
            video_features,
            timestep=t_action,
        )
        
        # Action loss
        loss_action = F.mse_loss(v_pred_action, target_v_action)
        
        # 7. Total loss
        total_loss = loss_video + self.action_loss_weight * loss_action
        
        # 8. Logging
        output_batch = {
            "video_loss": loss_video.detach(),
            "action_loss": loss_action.detach(),
            "total_loss": total_loss.detach(),
        }
        
        if iteration % self._action_head_log_every == 0:
            print(
                f"[joint] iter={iteration} "
                f"video_loss={loss_video.item():.6f} "
                f"action_loss={loss_action.item():.6f} "
                f"total={total_loss.item():.6f}"
            )
        
        return output_batch, total_loss
    
    @torch.no_grad()
    def infer_joint(
        self,
        data_batch: dict,
        num_inference_steps: int = 20,
        guidance: float = 1.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Joint inference: Video and Action denoise simultaneously.
        
        Similar to FastWAM's infer_joint.
        """
        # 1. Get initial latents and condition
        _, x0_video, condition = self.get_data_and_condition(data_batch)
        crossattn_emb = data_batch.get("t5_text_embeddings")
        B = x0_video.shape[0]
        device = x0_video.device
        
        # 2. Initialize noises
        latents_video = torch.randn_like(x0_video)
        latents_action = torch.randn(B, 64, 16, device=device, dtype=x0_video.dtype)
        
        # 3. Build inference schedule
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)[:-1]
        
        # 4. Denoising loop
        for i in range(num_inference_steps):
            t = timesteps[i]
            t_batch = torch.full((B,), t, device=device)
            
            # Video prediction (conditioned on current noisy action)
            v_video = self.denoise(
                noise=torch.randn_like(latents_video),  # Not used in inference
                xt_B_C_T_H_W=latents_video,
                timesteps_B_T=t_batch,
                condition=condition,
                action=latents_action,  # Current noisy action!
                crossattn_emb=crossattn_emb,
            )
            
            # Action prediction (conditioned on video features)
            # Note: In practice, you'd extract video features here
            # For simplicity, we use the action head directly
            v_action = self.action_head(
                latents_action,
                video_tokens=None,  # Would need to extract from video
                timestep=t_batch,
            )
            
            # Update latents
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            latents_video = latents_video - dt * v_video
            latents_action = latents_action - dt * v_action.unsqueeze(-1)
        
        return {
            "video": latents_video,
            "action": latents_action,
        }
