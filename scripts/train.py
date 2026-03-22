# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""
训练入口包装：直接转发给 cosmos_oss.scripts.train。

用法（从仓库根目录运行）:
    python -m scripts.train --config=configs/config.py -- experiment=my_action_experiment

或使用 torchrun:
    torchrun --nproc_per_node=1 -m scripts.train --config=configs/config.py -- experiment=my_action_experiment
"""

if __name__ == "__main__":
    from cosmos_oss.scripts.train import main

    main()
