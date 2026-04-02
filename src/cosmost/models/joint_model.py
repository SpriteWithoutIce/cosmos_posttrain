"""
Joint Video-Action Generation Model.
Similar to FastWAM but simplified for Cosmos-based training.

Key features:
1. Video generation with optional action condition (via timestep embedding)
2. Action prediction from video conditional frames
3. Joint training with shared gradient
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .dit import VideoDiT
from .action_head import ActionHead
from ..schedulers.rectified_flow import RectifiedFlow


class JointVideoActionModel(nn.Module):
    """
    Joint model for video and action generation.
    
    Architecture:
    - VideoDiT: generates video, can be conditioned on action via timestep
    - ActionHead: predicts actions from video features
    
    Training:
    - Video uses GT action as timestep condition
    - Action uses video conditional frames as input
    - Joint loss: video_loss + action_loss_weight * action_loss
    """
    
    def __init__(
        self,
        # Video DiT config
        video_dit_config: Dict,
        # Action head config
        action_head_enabled: bool = True,
        action_head_cfg: Optional[Dict] = None,
        action_loss_weight: float = 1.0,
        action_stop_gradient: bool = False,
        # Training config
        num_conditional_frames: int = 4,
    ):
        super().__init__()
        
        self.action_head_enabled = action_head_enabled
        self.action_loss_weight = action_loss_weight
        self.action_stop_gradient = action_stop_gradient
        self.num_conditional_frames = num_conditional_frames
        
        # Video DiT
        self.video_dit = VideoDiT(**video_dit_config)
        
        # Action head
        if action_head_enabled:
            cfg = dict(action_head_cfg or {})
            cfg["n_blocks"] = video_dit_config.get("num_blocks", 28)
            cfg["video_hidden_dim"] = video_dit_config.get("model_channels", 2048)
            self.action_head = ActionHead(**cfg)
        else:
            self.action_head = None
        
        # Rectified flow scheduler
        self.rectified_flow = RectifiedFlow(
            velocity_field=self.video_dit,
            train_time_distribution="logitnormal",
            shift=5.0,
        )
        
        # Logging
        self._log_every = int(os.environ.get("ACTION_HEAD_LOG_EVERY", "1"))
    
    def _extract_conditional_video_features(
        self,
        hidden_states: List[torch.Tensor],
        num_cond_frames: int,
    ) -> List[torch.Tensor]:
        """
        Extract features from conditional frames only.
        
        Args:
            hidden_states: List of [B, T*H*W, D] from each DiT layer
            num_cond_frames: number of conditional frames to extract
            
        Returns:
            List of [B, num_cond_frames*H*W, D]
        """
        extracted = []
        for hidden in hidden_states:
            B, N, D = hidden.shape
            # Assume T=12 (4 cond + 8 pred), H*W=60*80/4=1200
            # Need to infer H, W from N and num_cond_frames
            # N = T * H * W = 12 * H * W
            # So H * W = N // 12
            
            # For simplicity, assume we know the structure
            # hidden: [B, T*H*W, D]
            # We want: [B, num_cond_frames*H*W, D]
            
            total_frames = 12  # This should match your data
            tokens_per_frame = N // total_frames
            
            # Reshape to [B, T, H*W, D]
            hidden = hidden.view(B, total_frames, tokens_per_frame, D)
            
            # Take conditional frames
            hidden_cond = hidden[:, :num_cond_frames]  # [B, num_cond, H*W, D]
            
            # Reshape back
            hidden_cond = hidden_cond.reshape(B, num_cond_frames * tokens_per_frame, D)
            extracted.append(hidden_cond)
        
        return extracted
    
    def forward_video(
        self,
        x0_video: torch.Tensor,
        noise_video: torch.Tensor,
        t_video: torch.Tensor,
        crossattn_emb: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass for video generation.
        
        Args:
            x0_video: [B, C, T, H, W] GT video
            noise_video: [B, C, T, H, W] noise
            t_video: [B] timestep
            crossattn_emb: [B, L, D] text embeddings
            action: [B, 64, 16] action condition (optional)
            return_hidden: whether to return hidden states
            
        Returns:
            pred_velocity: [B, C, T, H, W]
            hidden_states: list of hidden states (if return_hidden=True)
        """
        # Get interpolation
        xt_video, target_v_video = self.rectified_flow.get_interpolation(
            noise_video, x0_video, t_video
        )
        
        # Forward through DiT
        if return_hidden:
            pred_v_video, hidden_states = self.video_dit(
                xt_video,
                t_video,
                crossattn_emb,
                action=action,
                return_hidden=True,
            )
            return pred_v_video, target_v_video, hidden_states
        else:
            pred_v_video = self.video_dit(
                xt_video,
                t_video,
                crossattn_emb,
                action=action,
            )
            return pred_v_video, target_v_video, None
    
    def forward_action(
        self,
        x0_action: torch.Tensor,
        noise_action: torch.Tensor,
        t_action: torch.Tensor,
        video_hidden_states: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for action prediction.
        
        Args:
            x0_action: [B, 64, 16] GT actions
            noise_action: [B, 64, 16] noise
            t_action: [B] timestep
            video_hidden_states: List of [B, N, D] from video DiT
            
        Returns:
            pred_velocity: [B, 64, 16]
            target_velocity: [B, 64, 16]
        """
        # Get interpolation
        xt_action = t_action.view(-1, 1, 1) * x0_action + (1 - t_action.view(-1, 1, 1)) * noise_action
        target_v_action = x0_action - noise_action
        
        # Extract conditional frames from video
        video_features = self._extract_conditional_video_features(
            video_hidden_states,
            self.num_conditional_frames,
        )
        
        # Forward through action head
        pred_v_action = self.action_head(xt_action, video_features, t_action)
        
        return pred_v_action, target_v_action
    
    def training_step(
        self,
        data_batch: Dict[str, torch.Tensor],
        iteration: int = 0,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Joint training step.
        
        Training process:
        1. Sample timesteps for video and action
        2. Add noise to video and action
        3. Video forward with GT action condition -> video_loss
        4. Action forward with video features -> action_loss
        5. Total loss = video_loss + action_loss_weight * action_loss
        """
        # Get data
        x0_video = data_batch["video"]
        B = x0_video.shape[0]
        device = x0_video.device
        
        # Text embeddings
        crossattn_emb = data_batch.get("t5_text_embeddings")
        
        # Sample timesteps
        t_video = self.rectified_flow.sample_train_time(B).to(device)
        t_action = torch.rand(B, device=device)
        
        # Prepare noises
        noise_video = torch.randn_like(x0_video)
        
        # Video forward with action condition
        if self.action_head_enabled and "actions" in data_batch:
            actions_gt = data_batch["actions"].to(device).float()
            noise_action = torch.randn_like(actions_gt)
            
            # Use GT action to condition video (if not stopping gradient)
            action_for_video = actions_gt if not self.action_stop_gradient else actions_gt.detach()
            
            pred_v_video, target_v_video, hidden_states = self.forward_video(
                x0_video,
                noise_video,
                t_video,
                crossattn_emb,
                action=action_for_video,
                return_hidden=True,
            )
            
            # Video loss
            loss_video = F.mse_loss(pred_v_video, target_v_video)
            
            # Action forward
            pred_v_action, target_v_action = self.forward_action(
                actions_gt,
                noise_action,
                t_action,
                hidden_states,
            )
            
            # Action loss
            loss_action = F.mse_loss(pred_v_action, target_v_action)
            
            # Total loss
            total_loss = loss_video + self.action_loss_weight * loss_action
            
            # Logging
            log_dict = {
                "loss": total_loss,
                "video_loss": loss_video.detach(),
                "action_loss": loss_action.detach(),
            }
            
            if iteration % self._log_every == 0:
                print(
                    f"[iter {iteration}] "
                    f"video_loss={loss_video.item():.4f} "
                    f"action_loss={loss_action.item():.4f} "
                    f"total={total_loss.item():.4f}"
                )
        else:
            # Video-only training
            pred_v_video, target_v_video, _ = self.forward_video(
                x0_video,
                noise_video,
                t_video,
                crossattn_emb,
                action=None,
            )
            loss_video = F.mse_loss(pred_v_video, target_v_video)
            total_loss = loss_video
            
            log_dict = {"loss": total_loss, "video_loss": loss_video.detach()}
            
            if iteration % self._log_every == 0:
                print(f"[iter {iteration}] video_loss={loss_video.item():.4f}")
        
        return log_dict, total_loss
    
    @torch.no_grad()
    def sample(
        self,
        data_batch: Dict[str, torch.Tensor],
        num_steps: int = 20,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference: generate video and action jointly.
        
        Args:
            data_batch: should contain text embeddings
            num_steps: number of denoising steps
            
        Returns:
            dict with "video" and "action" tensors
        """
        B = data_batch.get("t5_text_embeddings", torch.randn(1, 1, 1024)).shape[0]
        device = data_batch.get("t5_text_embeddings", torch.randn(1, 1, 1024)).device
        
        crossattn_emb = data_batch.get("t5_text_embeddings")
        
        # Initialize from noise
        latents_video = torch.randn(B, 16, 12, 60, 80, device=device)  # Adjust dimensions as needed
        latents_action = torch.randn(B, 64, 16, device=device)
        
        # Denoising loop
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)[:-1]
        
        for i in range(num_steps):
            t = timesteps[i]
            t_batch = torch.full((B,), t, device=device)
            
            # Video prediction (conditioned on current noisy action)
            v_video = self.video_dit(
                latents_video,
                t_batch,
                crossattn_emb,
                action=latents_action,
            )
            
            # Action prediction (simplified - would need hidden states in practice)
            # For inference, we can just run action head with zeros or cached features
            # This is a simplified version
            
            # Update latents (Euler method)
            dt = 1.0 / num_steps
            latents_video = latents_video - dt * v_video
            # For action, we'd need to properly compute velocity
            # This is simplified
        
        return {
            "video": latents_video,
            "action": latents_action,
        }
    
    def save_checkpoint(self, path: str, optimizer=None, iteration: int = 0):
        """Save model checkpoint."""
        checkpoint = {
            "model": self.state_dict(),
            "iteration": iteration,
        }
        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str, optimizer=None, strict: bool = True):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        self.load_state_dict(checkpoint["model"], strict=strict)
        
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        
        iteration = checkpoint.get("iteration", 0)
        print(f"Loaded checkpoint from {path} (iteration {iteration})")
        return iteration
