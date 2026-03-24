# RobotWin Deploy README

这个目录提供了一个完整的 `client <-> server <-> model` 评测链路，用于在 RoboTwin 仿真环境里评估当前 Cosmos 后训练模型。

## 1. 目录文件

- `cosmos_robotwin_server.py`  
  加载视频模型 + action head + VAE，提供 websocket 推理服务。
- `websocket_policy_server.py`  
  通用 websocket 服务封装（msgpack 二进制协议）。
- `eval_polict_client_openpi.py`  
  RoboTwin 评测客户端（参考 openpi 风格），负责跑任务、采样 obs、请求 server、执行动作、统计成功率。
- `websocket_client_policy.py`  
  轻量 websocket client。
- `launch_server.sh` / `launch_client.sh`  
  启动脚本。
- `export_ema_bf16.py`  
  DCP/ckpt 导出 `*_ema_bf16.pt`。

---

## 2. 整体调用链

1. 客户端从 RoboTwin 环境读取当前观测（主视角图像 + 当前 state）。
2. 客户端调用 server `infer`，发送一个 obs。
3. server 将图像编码成 latent，构造 4 帧条件 latent + 文本 embedding + 当前 state，调用模型采样。
4. server 用 `delta_v + state` 走 action head，返回 `action: [64, 16]`。
5. 客户端按顺序执行 64 步 16 维动作。
6. 客户端每 2 个执行步采一个关键帧，执行完后把关键帧序列用 `compute_kv_cache=True` 发回 server。
7. server 把关键帧再次时间降采样（`obs_seq[::2]`），编码进 latent 历史，供下一次推理使用。

---

## 3. Client 侧：拿到了什么，发送了什么

`eval_polict_client_openpi.py` 的 `format_obs()` 当前发送字段是：

```python
{
  "observation.images.cam_high": <H,W,3 uint8/float>,
  "observation.state": <16-dim float32>,
  "task": <str prompt>
}
```

关键点：

- 只使用主视角 `cam_high`，不再发送左右腕相机。
- `state` 使用“当前 obs 的 state”：
  - 优先 `observation["joint_action"]["vector"]`
  - 若不存在则回退为双臂 endpose 拼接。
- 每次调用 `model.infer({"obs": obs_dict, "prompt": prompt, "task_name": task_name})` 时，都是当前时刻最新 state。
- 执行动作时不做 `add_init_pose`，默认 server 返回的 16 维就是可直接执行的真实 action。

### Client 发送的三类请求

1) 重置：

```python
{"reset": True, "prompt": prompt, "task_name": task_name}
```

2) 单步推理：

```python
{"obs": obs_dict, "prompt": prompt, "task_name": task_name}
```

3) 回传关键帧更新历史：

```python
{
  "obs": [obs_k1, obs_k2, ...],
  "compute_kv_cache": True,
  "prompt": prompt,
  "task_name": task_name
}
```

---

## 4. Server 侧：处理流程和模型调用

`cosmos_robotwin_server.py` 的 `infer(payload)` 逻辑：

### A) reset 分支

- 命中 `payload["reset"] == True` 时：
  - 清空 `raw_obs_history / raw_state_history / latent_history / latent_state_history`
  - `step_id = 0`

### B) compute_kv_cache 分支

- 命中 `payload["compute_kv_cache"] == True` 时：
  - 取 `payload["obs"]`（list 或单个 dict）
  - 对序列做 `sampled = obs_seq[::2]`（再降采样一次）
  - 每帧提取：
    - `cam_high`
    - `observation.state`
  - 图像经 VAE 编码为单帧 latent `[16, H', W']`
  - 追加到 `latent_history`，同时 state 追加到 `latent_state_history`

### C) 正常 infer 分支

1. 解析当前 obs（只需要 `cam_high + state`），更新 `latest_state`。  
2. 构造条件 latent：
   - 从 `latent_history` 取最近 4 帧；
   - 若不足 4 帧，复制最后一帧补齐；
   - 得到 `cond_latent: [1, 16, 4, H', W']`。
3. 构造模型输入：
   - `pred_placeholder: [1, 16, 8, H', W']`（未来帧占位）
   - `video = concat(cond, pred)` -> `[1, 16, 12, H', W']`
   - `t5_text_embeddings`：从 pkl 先按 `task_name` 精确匹配，再 `prompt`，再模糊匹配。
   - `states`：当前 state（可选 qnorm）`[1, 16]`。
4. 调用 backbone：
   - `latents = model.generate_samples_from_batch(batch, ...)`
5. 提取动作条件：
   - `pred_v = latents[:, :, 4:12]`
   - `prev_v = latents[:, :, 3:11]`
   - `delta_v = rearrange(pred_v - prev_v, "b c t h w -> b t c h w")`
6. 调用 action head：
   - `z_action = zeros([1, 64, 16])`
   - `action = model.action_head(z_action, delta_v, state_t, timestep=0)[0]`
   - 输出 `action: [64, 16]`
7. 返回：

```python
{"action": action_np}
```

---

## 5. 维度和字段总结

- Client -> Server `obs`：
  - `observation.images.cam_high`: `[H, W, 3]`
  - `observation.state`: `[16]`
- VAE 编码后单帧 latent：
  - `[16, H', W']`
- 条件 latent：
  - `[1, 16, 4, H', W']`
- 模型 video 输入：
  - `[1, 16, 12, H', W']`（4 cond + 8 pred）
- Action head 输出：
  - `[64, 16]`（64 个连续动作步，每步 16 维）

---

## 6. 启动方法

## 6.1 启动 server

```bash
bash robotwin_deploy/launch_server.sh
```

常用环境变量（可选覆盖）：

- `VIDEO_CKPT`
- `ACTION_HEAD_CKPT`
- `VAE_PATH`
- `TEXT_EMB_PT`
- `STATS_JSON`
- `HOST` / `PORT`

## 6.2 启动 client

```bash
ROBOTWIN_ROOT=/root/linyihan/RoboTwin \
CLIENT_CONFIG=/root/linyihan/cosmos_posttrain/policy/ACT/deploy_policy.yml \
TASK_NAME=adjust_bottle \
TASK_CONFIG=demo_clean \
HOST=127.0.0.1 \
PORT=8000 \
bash robotwin_deploy/launch_client.sh
```

也可以直接运行：

```bash
ROBOTWIN_ROOT=/root/linyihan/RoboTwin \
python -m robotwin_deploy.eval_polict_client_openpi \
  --config /root/linyihan/cosmos_posttrain/policy/ACT/deploy_policy.yml \
  --tasks adjust_bottle \
  --host 127.0.0.1 \
  --port 8000 \
  --test_num 100 \
  --save_root ./results/cosmos_robotwin \
  --overrides --task_config demo_clean --seed 0
```

---

## 7. 你当前关注的三条约束（已对齐）

1. 只用主视角图像（`cam_high`）。  
2. 每次推理使用当前 obs 的 state。  
3. 动作输出 16 维直接执行，不做 init pose add/重构。  

---

## 8. 一个重要时序细节

当前实现下：

- 客户端每 2 步采一次关键帧；
- server 在 `compute_kv_cache` 中又做了一次 `obs_seq[::2]`。

等价于“最终写入 latent 历史”的频率约为每 4 个执行动作步 1 帧。  
如果你希望严格“每 2 步写 1 帧 latent 历史”，可以把 server 里的 `obs_seq[::2]` 去掉。

