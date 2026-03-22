# Robot Post-Training with Cosmos-Predict2.5 + Action Expert

基于 NVIDIA Cosmos-Predict2.5 Video2World 后训练，接入自定义 action expert 的独立训练仓库。

## 目录结构

```
robot_posttrain/
├── README.md
├── pyproject.toml
├── export_env.sh
├── train.sh
├── configs/
│   ├── config.py                          # 基础配置入口（action-conditioned）
│   └── experiments/
│       └── my_action_experiment.py         # 你的实验配置
├── datasets/
│   └── lerobot_latent_dataset.py           # 继承 LatentLeRobotDataset（不做归一化）
└── train.py                               # 训练入口（直接调用 cosmos_oss.scripts.train）
```

## 依赖说明

本仓库**不包含** cosmos-predict2.5 源码，只通过 `cosmos-oss` 包依赖其训练框架。
请先安装 `cosmos-predict2.5` 环境（参考 [官方文档](https://github.com/nvidia-cosmos/cosmos-predict2.5)），
本仓库的 `pyproject.toml` 已声明对 `cosmos-oss` 的依赖。

## 快速开始

### 1. 安装依赖

```bash
# 在 cosmos-predict2.5 环境中
pip install -e .

# 或独立环境
uv sync --extra=cu128
```

### 2. 设置环境变量

```bash
source export_env.sh
```

主要变量：
- `COSMOS_PT_CKPT` — post-train checkpoint 路径（默认 `/home/jwhe/linyihan/cosmos/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt`）
- `COSMOS_TOKENIZER` — VAE tokenizer 路径（默认 `/home/jwhe/linyihan/cosmos/tokenizer.pth`）
- `COSMOS_MEAN_STD` — VAE mean/std 路径（需确认，存在则使用）
- `DATASET_ROOT` — 数据集根目录（默认 `/home/jwhe/linyihan/datasets/lerobot_robotwin_eef_clean_50`）

### 3. 运行训练

```bash
bash train.sh
```

## 配置说明

所有关键配置集中在 `configs/experiments/my_action_experiment.py`：

| 配置项 | 说明 |
|---|---|
| `checkpoint.load_path` | post-train ckpt（你的 3.8GB 文件） |
| `model.config.state_t` | latent video 总帧数（= cond帧 + 预测帧） |
| `model.config.action_dim` | 动作维度（你的数据集为 16） |
| `dataloader_train` | 数据集路径、视频 keys、batch_size 等 |

## 与官方 action-conditioned 的区别

官方示例使用 `bridge_13frame_480_640_train`（BridgeData V2），
你的数据集使用 `LatentLeRobotDataset`（LeRobot 格式），两者的：

- 数据读取接口不同
- action 维度不同（16 vs 7）
- 视频 key 结构不同（多视角 tshape）

`lerobot_latent_dataset.py` 中已针对你的 `robotwin_tshape` 数据做了
`_action_post_process` / `_state_post_process` 处理，不做归一化（你说的需求）。
