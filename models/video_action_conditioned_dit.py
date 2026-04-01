from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.amp as amp
from hydra.core.config_store import ConfigStore
from torch import nn
from einops import rearrange

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import Attention, Block, MiniTrainDIT, SACConfig, VideoSize
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


class MaskedSelfAttention(Attention):
    def compute_attention(
        self,
        q,
        k,
        v,
        video_size: Optional[VideoSize] = None,
        kv_cache_cfg=None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        del kv_cache_cfg
        additional_args = {}
        if self.backend == "torch":
            additional_args["attn_mask"] = attn_mask
        elif attn_mask is not None:
            raise ValueError(f"Masked self attention requires torch backend, got {self.backend}")
        result = self.attn_op(q, k, v, **additional_args)
        return self.output_dropout(self.output_proj(result))

    def forward(
        self,
        x,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
        video_size: Optional[VideoSize] = None,
        kv_cache_cfg=None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
        return self.compute_attention(q, k, v, video_size=video_size, kv_cache_cfg=kv_cache_cfg, attn_mask=attn_mask)


class AsymmetricConditionBlock(Block):
    def __init__(self, *args, backend: str = "torch", use_wan_fp32_strategy: bool = False, **kwargs):
        super().__init__(*args, backend=backend, use_wan_fp32_strategy=use_wan_fp32_strategy, **kwargs)
        self.self_attn = MaskedSelfAttention(
            self.x_dim,
            None,
            self.self_attn.n_heads,
            self.self_attn.head_dim,
            qkv_format="bshd",
            backend=backend,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
        kv_cache_cfg=None,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if self.use_adaln_lora:
                shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                    self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
                shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                    self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
                shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                    self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
            else:
                shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = self.adaln_modulation_self_attn(
                    emb_B_T_D
                ).chunk(3, dim=-1)
                shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                    self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
                )
                shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        def _fn(_x_B_T_H_W_D, _norm_layer, _scale_B_T_1_1_D, _shift_B_T_1_1_D):
            return _norm_layer(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

        shift_self_attn_B_T_1_1_D = rearrange(shift_self_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        scale_self_attn_B_T_1_1_D = rearrange(scale_self_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        gate_self_attn_B_T_1_1_D = rearrange(gate_self_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        shift_cross_attn_B_T_1_1_D = rearrange(shift_cross_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        scale_cross_attn_B_T_1_1_D = rearrange(scale_cross_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        gate_cross_attn_B_T_1_1_D = rearrange(gate_cross_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        shift_mlp_B_T_1_1_D = rearrange(shift_mlp_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        scale_mlp_B_T_1_1_D = rearrange(scale_mlp_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        gate_mlp_B_T_1_1_D = rearrange(gate_mlp_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

        B, T, H, W, _ = x_B_T_H_W_D.shape
        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_self_attn,
            scale_self_attn_B_T_1_1_D,
            shift_self_attn_B_T_1_1_D,
        )
        video_size = VideoSize(T=T, H=H, W=W)
        result_B_T_H_W_D = rearrange(
            self.self_attn(
                rearrange(normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                None,
                rope_emb=rope_emb_L_1_1_D,
                video_size=video_size,
                kv_cache_cfg=kv_cache_cfg,
                attn_mask=self_attn_mask,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D * result_B_T_H_W_D

        normalized_cross_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_cross_attn,
            scale_cross_attn_B_T_1_1_D,
            shift_cross_attn_B_T_1_1_D,
        )
        result_cross_B_T_H_W_D = rearrange(
            self.cross_attn(
                rearrange(normalized_cross_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                crossattn_emb,
                rope_emb=rope_emb_L_1_1_D,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = result_cross_B_T_H_W_D * gate_cross_attn_B_T_1_1_D + x_B_T_H_W_D

        normalized_mlp_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_mlp,
            scale_mlp_B_T_1_1_D,
            shift_mlp_B_T_1_1_D,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_B_T_1_1_D * self.mlp(normalized_mlp_B_T_H_W_D)
        return x_B_T_H_W_D


class AsymmetricConditionDiT(MiniTrainDIT):
    supports_action_conditioning: bool = False

    def __init__(self, *args, **kwargs):
        backend = kwargs.get("atten_backend", "torch")
        crossattn_emb_channels = int(kwargs.get("crossattn_emb_channels", 1024))
        use_wan_fp32_strategy = bool(kwargs.get("use_wan_fp32_strategy", False))
        mlp_ratio = float(kwargs.get("mlp_ratio", 4.0))
        sac_config = kwargs.get("sac_config", SACConfig())
        super().__init__(*args, **kwargs)
        self.blocks = nn.ModuleList(
            [
                AsymmetricConditionBlock(
                    x_dim=self.model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=self.use_adaln_lora,
                    adaln_lora_dim=self.adaln_lora_dim,
                    backend=backend,
                    image_context_dim=None if self.extra_image_context_dim is None else self.model_channels,
                    use_wan_fp32_strategy=use_wan_fp32_strategy,
                )
                for _ in range(len(self.blocks))
            ]
        )
        for block in self.blocks:
            block.init_weights()
        self.enable_selective_checkpoint(sac_config, self.blocks)

    def _build_self_attention_mask(
        self,
        batch_size: int,
        num_frames: int,
        h: int,
        w: int,
        num_conditional_frames_B: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if num_conditional_frames_B is None or num_frames <= 1:
            return None
        tokens_per_frame = h * w
        total_tokens = num_frames * tokens_per_frame
        mask = torch.zeros((batch_size, 1, total_tokens, total_tokens), device=device, dtype=dtype)
        neg_inf = torch.finfo(dtype).min
        for b in range(batch_size):
            cond_frames = int(num_conditional_frames_B[b].item())
            cond_tokens = max(0, min(num_frames, cond_frames)) * tokens_per_frame
            if cond_tokens <= 0 or cond_tokens >= total_tokens:
                continue
            mask[b, :, :cond_tokens, cond_tokens:] = neg_inf
        return mask

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
        num_conditional_frames_B: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        del kwargs
        assert isinstance(data_type, DataType), f"Expected DataType, got {type(data_type)}."
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
        )
        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)
        context_input = crossattn_emb if img_context_emb is None else (crossattn_emb, self.img_context_proj(img_context_emb))

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        B, T, H, W, _ = x_B_T_H_W_D.shape
        self_attn_mask = self._build_self_attention_mask(
            batch_size=B,
            num_frames=T,
            h=H,
            w=W,
            num_conditional_frames_B=num_conditional_frames_B,
            device=x_B_T_H_W_D.device,
            dtype=x_B_T_H_W_D.dtype,
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
                self_attn_mask=self_attn_mask,
            )
            if intermediate_feature_ids and i in intermediate_feature_ids:
                intermediate_features_outputs.append(x_B_T_H_W_D.reshape(x_B_T_H_W_D.shape[0], -1, x_B_T_H_W_D.shape[-1]))

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        if intermediate_feature_ids:
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
    cs.store(
        group="net",
        package="model.config.net",
        name="cosmos_v1_2B_asymmetric_conditioned",
        node=L(AsymmetricConditionDiT)(
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
            atten_backend="torch",
            extra_per_block_abs_pos_emb=False,
            rope_h_extrapolation_ratio=1.0,
            rope_w_extrapolation_ratio=1.0,
            rope_t_extrapolation_ratio=1.0,
            sac_config=SACConfig(),
        ),
    )


__all__ = [
    "ActionTimestepConditionedDiT",
    "AsymmetricConditionDiT",
    "COSMOS_V1_2B_ACTION_TIMESTEP_NET",
    "register_local_video_nets",
]
