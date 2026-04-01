from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.amp as amp
from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import MiniTrainDIT, SACConfig
from models.video_action_conditioning import build_video_action_conditioner


class ActionTimestepConditionedDiT(MiniTrainDIT):
    supports_action_conditioning: bool = True

    def __init__(
        self,
        *args,
        timestep_scale: float = 1.0,
        video_action_conditioner_type: str = "mlp",
        video_action_conditioner_cfg: dict | None = None,
        **kwargs,
    ):
        self.timestep_scale = float(timestep_scale)
        self.video_action_conditioner_type = str(video_action_conditioner_type)
        super().__init__(*args, **kwargs)
        conditioner_cfg = dict(video_action_conditioner_cfg or {})
        conditioner_cfg.setdefault("model_channels", self.model_channels)
        conditioner_cfg.setdefault("use_adaln_lora", self.use_adaln_lora)
        self.video_action_conditioner = build_video_action_conditioner(self.video_action_conditioner_type, **conditioner_cfg)

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        intermediate_feature_ids: Optional[List[int]] = None,
        img_context_emb: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        del kwargs
        assert isinstance(data_type, DataType), (
            f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        )

        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
        )

        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if img_context_emb is not None:
            assert self.extra_image_context_dim is not None, (
                "extra_image_context_dim must be set if img_context_emb is provided"
            )
            img_context_emb = self.img_context_proj(img_context_emb)
            context_input = (crossattn_emb, img_context_emb)
        else:
            context_input = crossattn_emb

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            timesteps_B_T = timesteps_B_T * self.timestep_scale
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            if self.video_action_conditioner is not None and action is not None:
                action_emb_B_D, action_emb_B_3D = self.video_action_conditioner(action)
                if action_emb_B_D is not None:
                    t_embedding_B_T_D = t_embedding_B_T_D + action_emb_B_D
                if adaln_lora_B_T_3D is not None and action_emb_B_3D is not None:
                    adaln_lora_B_T_3D = adaln_lora_B_T_3D + action_emb_B_3D
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        self.affline_scale_log_info = {"t_embedding_B_T_D": t_embedding_B_T_D.detach()}
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb

        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, (
                f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"
            )

        intermediate_features_outputs = []
        for i, block in enumerate(self.blocks):
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                context_input,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )
            if intermediate_feature_ids and i in intermediate_feature_ids:
                intermediate_features_outputs.append(x_B_T_H_W_D.reshape(x_B_T_H_W_D.shape[0], -1, x_B_T_H_W_D.shape[-1]))

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        if intermediate_feature_ids:
            if len(intermediate_features_outputs) != len(intermediate_feature_ids):
                log.warning(
                    f"Collected {len(intermediate_features_outputs)} intermediate features, "
                    f"but expected {len(intermediate_feature_ids)}. "
                    f"Requested IDs: {intermediate_feature_ids}"
                )
            return x_B_C_Tt_Hp_Wp, intermediate_features_outputs

        return x_B_C_Tt_Hp_Wp


COSMOS_V1_2B_ACTION_TIMESTEP_NET: LazyDict = L(ActionTimestepConditionedDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=1.0,
    sac_config=SACConfig(),
)


def register_local_video_nets() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="net",
        package="model.config.net",
        name="cosmos_v1_2B_action_timestep_conditioned",
        node=COSMOS_V1_2B_ACTION_TIMESTEP_NET,
    )


__all__ = [
    "ActionTimestepConditionedDiT",
    "COSMOS_V1_2B_ACTION_TIMESTEP_NET",
    "register_local_video_nets",
]
