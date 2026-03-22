# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""
训练入口包装：直接转发给 cosmos_oss.scripts.train。

用法（从仓库根目录运行）:
    python -m scripts.train --config=configs/config.py -- experiment=my_action_experiment

或使用 torchrun:
    torchrun --nproc_per_node=1 -m scripts.train --config=configs/config.py -- experiment=my_action_experiment
"""

import faulthandler
import os
import signal
import sys


def _enable_debug_stack_dump():
    # Enable traceback dump for hard hangs:
    #   kill -USR1 <pid>
    faulthandler.enable(file=sys.stderr, all_threads=True)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    print(
        f"[debug] faulthandler enabled (pid={os.getpid()}). "
        "Send SIGUSR1 to dump all thread stacks.",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    _enable_debug_stack_dump()
    from cosmos_oss.scripts.train import main

    main()
