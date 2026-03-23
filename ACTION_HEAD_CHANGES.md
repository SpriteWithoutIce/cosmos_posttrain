# Action Head 改动说明（cosmos_posttrain）

## 改动文件
- `models/action_head.py`
- `models/precomputed_latent.py`
- `configs/experiments/my_action_experiment.py`
- `export_env.sh`

## 核心实现
1. 新增 `ActionMIPHead`
- 8 个 block。
- 每个 block: `1x self-attn + 3x delta-v cross-attn + 1x state cross-attn + FFN`。
- causal mask 生效（动作序列自回归掩码）。
- state 分支加入手工 `sigma` 门控：`sigma = exp(-k * ||delta_v||)`。

2. 在 `PrecomputedLatentVideo2WorldModel` 接入联合训练
- 保留原 video loss（来自基类 `training_step`）。
- 读取 batch 中：
  - `actions: [B, 64, 16]`
  - `states: [B, 16]`
- 从模型输出 `output_batch["model_pred"]` 提取 velocity 预测（与 cosmos rectified-flow forward 对齐）。
- 空间池化得到每帧 16 维速度向量后，按 `4 cond + 8 pred` 计算 `delta_v`：
  - `delta_v[t] = v[t] - v[t-1]`，共 8 个。
- MIP 双前向：
  - `z1 = N(0,1)`
  - `z2 = 0.9 * action_gt + noise`
  - 两分支都回归 `action_gt`，MSE 等权平均。
- 总损失：
  - `total_loss = video_loss + action_loss_weight * action_loss`

3. checkpoint 兼容策略（保持 video ckpt 可直接复用）
- 重写 `state_dict()`：过滤 `action_head.*`，主 ckpt 不包含 action head 参数。
- 重写 `load_state_dict(strict=True)`：
  - 对非 `action_head.*` 仍严格检查；
  - 允许主 ckpt 缺少 action head 参数（避免破坏你现有 video ckpt 严格加载流程）。

4. action head 参数独立存储
- 新增配置：
  - `ACTION_HEAD_SAVE_EVERY`
  - `ACTION_HEAD_SAVE_DIR`
  - `ACTION_HEAD_LOAD_PATH`
- 训练中按周期保存：
  - `${ACTION_HEAD_SAVE_DIR}/action_head_iter_XXXXXXX.pt`
- 可单独加载 action head 参数，不影响 video ckpt。

5. action loss 日志
- 控制台周期打印：
  - `[action-head] iter=... video_loss=... action_loss_1=... action_loss_2=... action_loss=... total_loss=...`
- 若 wandb 已开启（online/offline 且 run 已初始化），同步记录：
  - `train/video_loss`
  - `train/action_loss_1`
  - `train/action_loss_2`
  - `train/action_loss`
  - `train/total_loss`
- 同时在 `output_batch` 写入：
  - `action_loss_1 / action_loss_2`
  - `action_loss / video_loss / total_loss`
  - `metrics/action_loss_1 / metrics/action_loss_2`
  - `metrics/action_loss / metrics/video_loss / metrics/total_loss`

6. action/state q01-q99 归一化
- 在 `LeRobotLatentDataset` 中优先读取全局 `stats.json`（`ACTION_STATE_GLOBAL_STATS_JSON`）：
  - `action.q01/q99`
  - `observation.state.q01/q99`
- 若未配置全局路径，则回退读取每个 task 的 `meta/stats.json`。
- 返回给模型的 `actions / states / states_seq` 会先映射到 `[-1, 1]`：
  - `norm = 2*(x-q01)/(q99-q01+eps)-1`
- 支持裁剪（默认 `[-1, 1]`）防止极端值放大 loss。
- 归一化由环境变量控制：
  - `ACTION_STATE_USE_QNORM`
  - `ACTION_STATE_NORM_CLIP`
  - `ACTION_STATE_GLOBAL_STATS_JSON`

7. Action Head 升级为 DiT 风格（参考 reasoningVLA）
- `models/action_head.py` 重构为 timestep-aware 的 DiT 样式：
  - sinusoidal timestep embedding + MLP
  - AdaLayerNorm 调制
  - 引入参考实现风格的 `action_encoder / state_encoder / action_decoder`
  - 每个 block: `1x self-attn + 3x delta cross-attn + 1x state cross-attn + FFN`
  - state 分支保留 sigma 门控
- 训练时使用 `output_batch["timesteps"]` 作为 action head 的 timestep 条件。
  （已进一步改为可独立采样，见第 8 条）

10. delta_v 保留空间高维（不再全局均值）
- 之前：`model_pred` 在 `H,W` 上做 `mean` 后得到 `[B,8,16]`。
- 现在：直接在 latent 空间算差分，得到 `[B,8,C,H,W]`，并展开为空间 token 做 cross-attn。
- 可选池化开关（防显存压力）：
  - `ACTION_HEAD_DELTA_POOL_H`
  - `ACTION_HEAD_DELTA_POOL_W`
  - 默认 `0` 表示不池化、完整保留空间分辨率。

8. 双时间步机制（按你的新需求）
- video 主分支：继续使用原始随机 RF 时间步训练 video loss。
- action 的 `delta_v` 分支：支持固定视频采样时间步 `ACTION_DELTA_VIDEO_T`（例如 0.5）：
  - 额外构造 fixed-t 的 `xt` 并调用一次 `denoise` 得到 velocity
  - 用该 velocity 计算 8 个 `delta_v`
  - 这条分支保留梯度，`action_loss` 仍可更新 video model 参数
- action head 的 timestep 与 video timestep 解耦：
  - `ACTION_HEAD_TIMESTEP_MODE=beta`：Beta 连续采样后离散化（默认）
  - `ACTION_HEAD_TIMESTEP_MODE=random`：独立均匀随机
  - `ACTION_HEAD_TIMESTEP_MODE=fixed`：固定 `ACTION_HEAD_FIXED_TIMESTEP`
- action 分支噪声轨迹更新为 Beta 时间控制：
  - `z2 = (1-t)*noise + t*action`
  - `t` 为 action head 独立采样时间（Beta/Random/Fixed）

9. 分组学习率（video / action expert 分开）
- 在 `PrecomputedLatentVideo2WorldModel.init_optimizer_scheduler` 中改为 2 个 param group：
  - group0: `self.net`（video 主干），使用原配置 `optimizer.lr`
  - group1: `self.action_head`，使用 `ACTION_HEAD_LR`（默认 `1e-4`）
- 两组共享 `weight_decay`、`betas`、`eps`，优化器类型沿用配置（`fusedadam` / `adamw`）。
- wandb 的 `optim/lr_0`、`optim/lr_1` 会分别显示两组学习率。

## 环境变量
在 `export_env.sh` 新增：
- `ACTION_HEAD_ENABLED`
- `ACTION_HEAD_LR`
- `ACTION_HEAD_LOSS_WEIGHT`
- `ACTION_HEAD_SAVE_EVERY`
- `ACTION_HEAD_SAVE_DIR`
- `ACTION_HEAD_LOAD_PATH`
- `ACTION_HEAD_LOG_EVERY`
- `ACTION_HEAD_WANDB_LOG`
- `ACTION_DELTA_VIDEO_T`
- `ACTION_HEAD_TIMESTEP_MODE`
- `ACTION_HEAD_FIXED_TIMESTEP`
- `ACTION_HEAD_NOISE_BETA_ALPHA`
- `ACTION_HEAD_NOISE_BETA_BETA`
- `ACTION_HEAD_NOISE_BETA_S`
- `ACTION_HEAD_DELTA_POOL_H`
- `ACTION_HEAD_DELTA_POOL_W`
- `ACTION_STATE_USE_QNORM`
- `ACTION_STATE_NORM_CLIP`
- `ACTION_STATE_GLOBAL_STATS_JSON`
