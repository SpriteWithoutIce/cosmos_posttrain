"""Legacy MIP (Multiple Importance Path) action head.

Wraps the original ActionMIPHead from models/action_head.py with the new
BaseActionHead interface so it can be used through the registry.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Beta

from models.action_head import ActionMIPHead
from models.action_heads import register_action_head
from models.action_heads.base import BaseActionHead


@register_action_head("mip")
class MIPActionHead(BaseActionHead):
    """MIP action head adapter — wraps the original ActionMIPHead.

    Training uses two forward passes (z1=noise, z2=mixed) and sums both MSE losses.
    Condition features are expected as delta-velocity tensors.
    """

    def __init__(
        self,
        action_dim: int = 16,
        num_actions: int = 64,
        d_model: int = 256,
        n_heads: int = 8,
        n_blocks: int = 8,
        delta_dim: int = 16,
        state_dim: int = 16,
        sigma_k: float = 4.0,
        dropout: float = 0.0,
        timestep_buckets: int = 1000,
        delta_spatial_pool_h: int = 0,
        delta_spatial_pool_w: int = 0,
        delta_height: int = 60,
        delta_width: int = 80,
        actions_per_latent: int = 8,
        # MIP training hyper-parameters
        mip_gt_mix: float = 0.9,
        timestep_mode: str = "beta",
        fixed_timestep: int = 0,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
        # state branch control
        use_state: bool = True,
        **_extra,
    ):
        super().__init__()
        self.use_state = use_state
        self.mip_gt_mix = float(mip_gt_mix)
        self.timestep_mode = str(timestep_mode)
        self.fixed_timestep = int(fixed_timestep)
        self._beta_dist = Beta(float(noise_beta_alpha), float(noise_beta_beta))
        self.noise_s = float(noise_s)

        self.inner = ActionMIPHead(
            action_dim=action_dim,
            num_actions=num_actions,
            d_model=d_model,
            n_heads=n_heads,
            n_blocks=n_blocks,
            delta_dim=delta_dim,
            state_dim=state_dim,
            sigma_k=sigma_k,
            dropout=dropout,
            timestep_buckets=timestep_buckets,
            delta_spatial_pool_h=delta_spatial_pool_h,
            delta_spatial_pool_w=delta_spatial_pool_w,
            delta_height=delta_height,
            delta_width=delta_width,
            actions_per_latent=actions_per_latent,
        )
        self.timestep_buckets = timestep_buckets
        self.action_dim = action_dim
        self.num_actions = num_actions

    # ------------------------------------------------------------------
    # Timestep sampling (moved from PrecomputedLatentVideo2WorldModel)
    # ------------------------------------------------------------------
    def _sample_timestep(self, batch_size: int, device: torch.device):
        buckets = self.timestep_buckets
        mode = self.timestep_mode.lower()
        if mode == "fixed":
            t = max(0, min(buckets - 1, self.fixed_timestep))
            t_disc = torch.full((batch_size,), t, device=device, dtype=torch.long)
            t_cont = t_disc.float() / max(buckets - 1, 1)
            return t_cont, t_disc
        if mode == "random":
            t_cont = torch.rand(batch_size, device=device, dtype=torch.float32)
            t_disc = torch.clamp((t_cont * buckets).long(), 0, buckets - 1)
            return t_cont, t_disc
        # beta
        sample = self._beta_dist.sample([batch_size]).to(device=device, dtype=torch.float32)
        t_cont = (self.noise_s - sample) / self.noise_s
        t_cont = torch.clamp(t_cont, 0.0, 1.0)
        t_disc = torch.clamp((t_cont * buckets).long(), 0, buckets - 1)
        return t_cont, t_disc

    # ------------------------------------------------------------------
    # BaseActionHead interface
    # ------------------------------------------------------------------
    def forward(
        self,
        z_action: Tensor,
        condition_features: Tensor,
        timestep: Tensor | None = None,
        state_vec: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        if state_vec is None or not self.use_state:
            state_vec = torch.zeros(z_action.shape[0], 16, device=z_action.device, dtype=z_action.dtype)
        return self.inner(z_action, condition_features, state_vec, timestep=timestep)

    def compute_loss(
        self,
        actions_gt: Tensor,
        condition_features: Tensor,
        state_vec: Tensor | None = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        bsz = actions_gt.shape[0]
        device = actions_gt.device

        t2_cont, t2_disc = self._sample_timestep(bsz, device)
        t1 = torch.zeros_like(t2_disc)

        noise = torch.randn_like(actions_gt)
        z1 = noise
        z2 = self.mip_gt_mix * actions_gt + (1.0 - self.mip_gt_mix) * noise

        pred1 = self.forward(z1, condition_features, timestep=t1, state_vec=state_vec)
        pred2 = self.forward(z2, condition_features, timestep=t2_disc, state_vec=state_vec)

        loss1 = F.mse_loss(pred1, actions_gt)
        loss2 = F.mse_loss(pred2, actions_gt)
        loss = loss1 + loss2

        return {
            "loss": loss,
            "action_loss_1": loss1.detach(),
            "action_loss_2": loss2.detach(),
            "action_timestep_mean": t2_cont.detach().mean(),
        }
