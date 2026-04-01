# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Post-training base config for action-conditioned Video2World.
# 继承自 cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py
# 只在当前目录重新定义 Experiment 注册入口，model / net / conditioner 等组件
# 全部复用官方 action-conditioned 体系。

from typing import Any, List

import attrs

from cosmos_predict2._src.imaginaire.config import Config as ImaginaireConfig
from cosmos_predict2._src.imaginaire.trainer import ImaginaireTrainer as Trainer
from cosmos_predict2._src.imaginaire.utils.config_helper import import_all_modules_from_package
from cosmos_predict2._src.predict2.action.configs.action_conditioned.conditioner import register_conditioner
from cosmos_predict2._src.predict2.action.configs.action_conditioned.data import register_training_and_val_data
from cosmos_predict2._src.predict2.action.configs.action_conditioned.model import register_model
from cosmos_predict2._src.predict2.action.configs.action_conditioned.net import register_net
from cosmos_predict2._src.predict2.configs.common.defaults.checkpoint import register_checkpoint
from cosmos_predict2._src.predict2.configs.common.defaults.ckpt_type import register_ckpt_type
from cosmos_predict2._src.predict2.configs.common.defaults.ema import register_ema
from cosmos_predict2._src.predict2.configs.common.defaults.optimizer import register_optimizer
from cosmos_predict2._src.predict2.configs.common.defaults.scheduler import register_scheduler
from cosmos_predict2._src.predict2.configs.common.defaults.tokenizer import register_tokenizer
from cosmos_predict2._src.predict2.configs.video2world.defaults.callbacks import register_callbacks
from cosmos_predict2._src.predict2.configs.video2world.defaults.net import register_net as register_video_net
from models.video_action_conditioned_dit import register_local_video_nets


@attrs.define(slots=False)
class Config(ImaginaireConfig):
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"data_train": "mock"},
            {"data_val": "mock"},
            {"optimizer": "fusedadamw"},
            {"scheduler": "lambdalinear"},
            {"model": "action_conditioned_video2world_fsdp_rectified_flow"},
            {"callbacks": "basic"},
            {"net": None},
            {"conditioner": "action_conditioned_video_conditioner"},
            {"ema": "power"},
            {"tokenizer": "wan2pt2_tokenizer"},
            {"checkpoint": "s3"},
            {"ckpt_type": "dummy"},
            {"experiment": None},
        ]
    )


def make_config() -> Config:
    c = Config(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    c.job.project = "cosmos_diffusion_v2"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = Trainer
    c.trainer.straggler_detection.enabled = False
    c.trainer.max_iter = 400_000
    c.trainer.logging_iter = 10
    c.trainer.validation_iter = 100
    c.trainer.run_validation = False
    c.trainer.callbacks = None

    # 注册官方组件
    register_optimizer()
    register_scheduler()
    register_model()
    register_callbacks()
    register_ema()
    register_tokenizer()
    register_checkpoint()
    register_ckpt_type()
    register_training_and_val_data()
    register_net()
    register_video_net()
    register_local_video_nets()
    register_conditioner()

    # 官方 experiment 列表（必须导入以便 Hydra 解析 experiment=...）
    import_all_modules_from_package("cosmos_predict2.experiments", reload=True)
    import_all_modules_from_package("cosmos_predict2._src.predict2.configs.video2world.experiment", reload=True)
    import_all_modules_from_package(
        "cosmos_predict2._src.predict2.action.configs.action_conditioned.experiment", reload=True
    )

    # ★ 你的自定义 experiment 放在这里 ★
    import_all_modules_from_package("configs.experiments", reload=True)

    return c
