# Joint Generation 实现指南 (Cross-Attention 版本)

## 1. 核心修改点

### 1.1 Video Network 修改

需要创建一个新的 DiT 类，支持将 action 作为 cross-attention 的 condition：

```python
# models/action_cross_attention_dit.py

from cosmos_predict2._src.predict2.networks.minimal_v4_dit import MiniTrainDIT, Block
from models.video_action_conditioned_dit import AsymmetricConditionBlock

class ActionCrossAttentionDiT(MiniTrainDIT):
    """
    DiT that uses action as cross-attention condition.
    Similar to FastWAM's video_expert.
    """
    supports_action_conditioning: bool = True
    
    def __init__(self, *args, action_embed_dim: int = 512, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Action embedding projection
        self.action_embed_dim = action_embed_dim
        self.action_proj = nn.Sequential(
            nn.Linear(64 * 16, action_embed_dim),  # 64 actions * 16 dim
            nn.LayerNorm(action_embed_dim),
            nn.GELU(),
        )
        
    def forward(self, x_B_C_T_H_W, timesteps_B_T, crossattn_emb, 
                action=None, **kwargs):
        """
        Args:
            action: [B, 64, 16] - action sequence
        """
        # Prepare embeddings (same as MiniTrainDIT)
        x_B_T_H_W_D, rope_emb, extra_pos = self.prepare_embedded_sequence(x_B_C_T_H_W)
        
        # Process cross-attention embeddings
        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)
        
        # Add action to cross-attention context
        if action is not None:
            B = action.shape[0]
            action_flat = rearrange(action, "b n d -> b (n d)")
            action_emb = self.action_proj(action_flat)  # [B, action_embed_dim]
            
            # Concatenate to text embeddings
            if crossattn_emb is not None:
                # Expand action to match sequence length for broadcasting
                action_emb = action_emb.unsqueeze(1)  # [B, 1, D]
                crossattn_emb = torch.cat([crossattn_emb, action_emb], dim=1)
        
        # Timestep embedding
        t_emb, adaln_lora = self.t_embedder(timesteps_B_T)
        t_emb = self.t_embedding_norm(t_emb)
        
        # Forward through blocks
        for block in self.blocks:
            x_B_T_H_W_D = block(
                x_B_T_H_W_D, t_emb, crossattn_emb, 
                rope_emb, adaln_lora, extra_pos
            )
        
        # Output
        output = self.final_layer(x_B_T_H_W_D, t_emb, adaln_lora)
        return self.unpatchify(output)
```

### 1.2 Model 修改

修改 `precomputed_latent.py` 中的 training_step：

```python
def training_step(self, data_batch, iteration):
    # 1. Get data
    x0_video = data_batch["video"]
    actions_gt = data_batch["actions"]  # [B, 64, 16]
    
    # 2. Sample timesteps
    t_video = sample_timestep(B)
    t_action = sample_timestep(B)
    
    # 3. Prepare noisy latents
    xt_video = add_noise(x0_video, t_video)
    xt_action = add_noise(actions_gt, t_action)
    
    # 4. Video forward with action cross-attention
    # Key: action is used as cross-attn condition, not timestep
    v_pred_video, hidden = self.net(
        xt_video,
        timesteps=t_video,
        crossattn_emb=text_emb,
        action=actions_gt,  # GT action for training
        collect_hidden=True,
    )
    
    # 5. Action forward with video features
    video_cond = extract_cond_frames(hidden, num_cond=4)
    v_pred_action = self.action_head(xt_action, video_cond, t_action)
    
    # 6. Joint loss
    loss = mse(v_pred_video, target_video) + mse(v_pred_action, target_action)
    return loss
```

### 1.3 Inference 修改

```python
@torch.no_grad()
def infer_joint(self, data_batch, num_steps=20):
    # Initialize noises
    video = torch.randn(B, C, T, H, W)
    action = torch.randn(B, 64, 16)
    
    for step in range(num_steps):
        t = get_timestep(step)
        
        # Video denoise (conditioned on current noisy action)
        v_video = self.net(
            video, t, 
            crossattn_emb=text_emb,
            action=action,  # Current noisy action
        )
        
        # Action denoise (conditioned on video)
        _, hidden = self.net(video, t, return_hidden=True)
        video_cond = extract_cond_frames(hidden, num_cond=4)
        v_action = self.action_head(action, video_cond, t)
        
        # Update
        video = video - dt * v_video
        action = action - dt * v_action
    
    return video, action
```

## 2. 与 FastWAM 的对比

| 方面 | FastWAM | 你的实现 (Cross-Attn) |
|------|---------|----------------------|
| Action→Video | Action 作为 context tokens，video 通过 cross-attn 看到 | 同上 |
| Video→Action | Video KV cache，action 通过 mixed attention 看到 | Video cond frames → Action head |
| 训练一致性 | GT action condition video | GT action condition video ✓ |
| 推理一致性 | Noisy action condition video | Noisy action condition video ✓ |

## 3. 关键优势

1. **训练/推理一致**：都使用 action condition video
2. **双向影响**：
   - Action → Video (通过 cross-attention)
   - Video → Action (通过 video features)
3. **Flexible**：Action 只看到 clean conditional frames

## 4. 实现建议

### 方案 A：最小修改（推荐先尝试）
保持现有 `ActionTimestepConditionedDiT`，但修改调用方式：

```python
# 不使用 action 作为 timestep，而是作为 extra context
class SimpleJointModel(PrecomputedLatentVideo2WorldModel):
    def training_step(self, data_batch, iteration):
        # Video 不用 action condition (保持原样)
        # Action 用 video hidden
        # 只优化 action loss，让 action head 学好
        # 然后再 joint training
        pass
```

### 方案 B：Full Cross-Attention（推荐）
创建新的 DiT class，参考上面的 `ActionCrossAttentionDiT`。

### 方案 C：Hybrid
使用现有 `AsymmetricConditionDiT` 但修改 cross-attention mask：
- 让 video tokens 只看到 text + action 的某些部分

## 5. 推荐路径

建议先实现 **方案 A**（最简单）：

1. 保持当前 video training（不用 action condition）
2. 但让 action head 使用 video 的 conditional frames
3. 先训练到收敛
4. 然后尝试 joint training（video + action loss）

如果 action loss 降不下来，再考虑 full cross-attention。

你想从哪个方案开始？
