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
- 从模型输出 `output_batch` 中提取 velocity 预测（优先 key: `vt_pred_B_C_T_H_W`）。
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

## 环境变量
在 `export_env.sh` 新增：
- `ACTION_HEAD_ENABLED`
- `ACTION_HEAD_LOSS_WEIGHT`
- `ACTION_HEAD_SAVE_EVERY`
- `ACTION_HEAD_SAVE_DIR`
- `ACTION_HEAD_LOAD_PATH`
