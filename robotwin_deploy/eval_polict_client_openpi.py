from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

try:
    from .robotwin_client import parse_action_steps
    from .websocket_client_policy import WebsocketClientPolicy
except ImportError:
    from robotwin_client import parse_action_steps
    from websocket_client_policy import WebsocketClientPolicy


def ensure_robowin_importable(robowin_root: str) -> None:
    root = Path(robowin_root).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)


def write_json(data: dict[str, Any], fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def class_decorator(task_name: str):
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def format_obs(observation: dict[str, Any], prompt: str) -> dict[str, Any]:
    return {
        "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"],
        "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
        "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],
        "observation.state": observation["joint_action"]["vector"],
        "task": prompt,
    }


def euler2quat(rx: float, ry: float, rz: float) -> np.ndarray:
    return R.from_euler("xyz", [rx, ry, rz], degrees=False).as_quat().astype(np.float64)


def add_eef_pose(new_pose: np.ndarray, init_pose: np.ndarray) -> np.ndarray:
    new_pose_r = R.from_quat(new_pose[3:7][None])
    init_pose_r = R.from_quat(init_pose[3:7][None])
    out_rot = (init_pose_r * new_pose_r).as_quat().reshape(-1)
    out_trans = new_pose[:3] + init_pose[:3]
    return np.concatenate([out_trans, out_rot, new_pose[7:8]])


def add_init_pose(new_pose: np.ndarray, init_pose: np.ndarray) -> np.ndarray:
    left_pose = add_eef_pose(new_pose[:8], init_pose[:8])
    right_pose = add_eef_pose(new_pose[8:], init_pose[8:])
    return np.concatenate([left_pose, right_pose])


def to_env_ee_action(raw_action_step: np.ndarray, init_eef_pose: np.ndarray) -> np.ndarray:
    ee_action = raw_action_step.reshape(-1)
    if ee_action.shape[0] == 14:
        return np.concatenate(
            [
                ee_action[:3],
                euler2quat(ee_action[3], ee_action[4], ee_action[5]),
                ee_action[6:10],
                euler2quat(ee_action[10], ee_action[11], ee_action[12]),
                ee_action[13:14],
            ]
        )
    if ee_action.shape[0] == 16:
        ee_action = add_init_pose(ee_action, init_eef_pose)
        return np.concatenate(
            [
                ee_action[:3],
                ee_action[3:7] / np.linalg.norm(ee_action[3:7]),
                ee_action[7:11],
                ee_action[11:15] / np.linalg.norm(ee_action[11:15]),
                ee_action[15:16],
            ]
        )
    raise NotImplementedError(f"Unsupported action dim: {ee_action.shape[0]}")


def normalize_setup_kwargs(setup_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(setup_kwargs or {})
    for reserved in ("now_ep_num", "seed", "is_test"):
        out.pop(reserved, None)
    random_setting = out.get("random_setting", None)
    if random_setting is None:
        out["random_setting"] = {}
    elif not isinstance(random_setting, dict):
        raise TypeError(f"random_setting must be dict or None, got {type(random_setting)}")
    return out


def setup_demo_safe(task_env: Any, *, now_ep_num: int, seed: int, is_test: bool, setup_kwargs: dict[str, Any]) -> None:
    kwargs = dict(setup_kwargs)
    kwargs["random_setting"] = kwargs.get("random_setting") or {}
    task_env.setup_demo(now_ep_num=now_ep_num, seed=seed, is_test=is_test, **kwargs)


def parse_override_pairs(pairs: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if len(pairs) % 2 != 0:
        raise ValueError("--overrides must be key value pairs.")
    for i in range(0, len(pairs), 2):
        key = pairs[i].lstrip("-")
        value_raw = pairs[i + 1]
        try:
            value = eval(value_raw)
        except Exception:
            value = value_raw
        out[key] = value
    return out


def eval_policy(
    task_env: Any,
    model: WebsocketClientPolicy,
    task_name: str,
    test_num: int,
    st_seed: int,
    save_root: str,
    setup_kwargs: dict[str, Any],
    clear_cache_freq: int = 1,
    expert_check: bool = True,
) -> tuple[int, int]:
    from envs.utils.create_actor import UnStableError

    task_env.suc = 0
    task_env.test_num = 0
    now_ep_num = 0
    now_seed = st_seed
    succ_seed = 0

    while succ_seed < test_num:
        if expert_check:
            try:
                setup_demo_safe(
                    task_env,
                    now_ep_num=now_ep_num,
                    seed=now_seed,
                    is_test=True,
                    setup_kwargs=setup_kwargs,
                )
                task_env.play_once()
                task_env.close_env()
            except UnStableError:
                task_env.close_env()
                now_seed += 1
                continue
            except Exception:
                task_env.close_env()
                traceback.print_exc()
                now_seed += 1
                continue

            if not (task_env.plan_success and task_env.check_success()):
                now_seed += 1
                continue

        succ_seed += 1
        setup_demo_safe(
            task_env,
            now_ep_num=now_ep_num,
            seed=now_seed,
            is_test=True,
            setup_kwargs=setup_kwargs,
        )
        prompt = task_env.get_instruction()
        model.infer({"reset": True, "prompt": prompt, "task_name": task_name})

        init_obs = task_env.get_obs()
        init_eef_pose = init_obs["endpose"]["left_endpose"] + [init_obs["endpose"]["left_gripper"]]
        init_eef_pose += init_obs["endpose"]["right_endpose"] + [init_obs["endpose"]["right_gripper"]]
        init_eef_pose = np.array(init_eef_pose, dtype=np.float64)

        while task_env.take_action_cnt < task_env.step_lim:
            first_obs = format_obs(task_env.get_obs(), prompt)
            ret = model.infer({"obs": first_obs, "prompt": prompt, "task_name": task_name})
            action_steps = parse_action_steps(ret["action"])
            key_frame_list: list[dict[str, Any]] = []

            for step_i, raw_step in enumerate(action_steps):
                env_action = to_env_ee_action(np.asarray(raw_step), init_eef_pose)
                task_env.take_action(env_action, action_type="ee")
                if (step_i + 1) % 2 == 0:
                    key_frame_list.append(format_obs(task_env.get_obs(), prompt))
                if task_env.eval_success:
                    break

            if key_frame_list:
                model.infer(
                    {
                        "obs": key_frame_list,
                        "compute_kv_cache": True,
                        "prompt": prompt,
                        "task_name": task_name,
                    }
                )
            if task_env.eval_success:
                break

        succ = bool(task_env.eval_success)
        if succ:
            task_env.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        task_env.close_env(clear_cache=((succ_seed + 1) % max(clear_cache_freq, 1) == 0))
        task_env.test_num += 1
        now_ep_num += 1
        now_seed += 1

        save_dir = Path(save_root) / f"stseed-{st_seed}" / "metrics" / task_name
        save_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            {
                "succ_num": float(task_env.suc),
                "total_num": float(task_env.test_num),
                "succ_rate": float(task_env.suc / max(task_env.test_num, 1)),
            },
            save_dir / "res.json",
        )
        print(
            f"{task_name}: success {task_env.suc}/{task_env.test_num} "
            f"({round(task_env.suc / max(task_env.test_num, 1) * 100, 1)}%), seed={now_seed}"
        )

    return now_seed, task_env.suc


def parse_args_and_config() -> dict[str, Any]:
    parser = argparse.ArgumentParser("RobotWin eval client (OpenPI-style) for cosmos server")
    parser.add_argument("--config", type=str, default="", help="Optional yaml config.")
    parser.add_argument("--overrides", nargs=argparse.REMAINDER, default=[])
    parser.add_argument("--robowin_root", type=str, default="")
    parser.add_argument("--task_name", type=str, default="")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--test_num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_root", type=str, default="./results/cosmos_robotwin")
    parser.add_argument("--clear_cache_freq", type=int, default=1)
    parser.add_argument("--expert_check", type=int, default=1, help="1/0")
    args = parser.parse_args()

    cfg: dict[str, Any] = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise TypeError(f"config must be dict yaml, got {type(loaded)}")
        cfg.update(loaded)

    cfg.update(parse_override_pairs(args.overrides))
    for k, v in vars(args).items():
        if k in ("config", "overrides"):
            continue
        if v not in (None, ""):
            cfg[k] = v

    if "robowin_root" not in cfg or not cfg["robowin_root"]:
        raise ValueError("--robowin_root is required (or provide in config).")
    if "task_name" not in cfg or not cfg["task_name"]:
        raise ValueError("--task_name is required (or provide in config).")
    return cfg


def main() -> None:
    cfg = parse_args_and_config()
    ensure_robowin_importable(cfg["robowin_root"])

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = cfg["task_name"]
    seed = int(cfg.get("seed", 0))
    test_num = int(cfg.get("test_num", 100))
    save_root = str(cfg.get("save_root", "./results/cosmos_robotwin"))
    st_seed = 10000 * (1 + seed)

    setup_kwargs = normalize_setup_kwargs(cfg.get("setup_kwargs", {}))
    if "random_setting" in cfg and "random_setting" not in setup_kwargs:
        setup_kwargs["random_setting"] = cfg["random_setting"]
    setup_kwargs = normalize_setup_kwargs(setup_kwargs)

    task_env = class_decorator(task_name)
    model = WebsocketClientPolicy(host=cfg.get("host", "127.0.0.1"), port=int(cfg.get("port", 8000)))

    _, suc_num = eval_policy(
        task_env=task_env,
        model=model,
        task_name=task_name,
        test_num=test_num,
        st_seed=st_seed,
        save_root=save_root,
        setup_kwargs=setup_kwargs,
        clear_cache_freq=int(cfg.get("clear_cache_freq", 1)),
        expert_check=bool(int(cfg.get("expert_check", 1))),
    )

    result_file = Path(save_root) / f"stseed-{st_seed}" / "metrics" / task_name / "_result.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"Timestamp: {current_time}\n")
        f.write(f"Task: {task_name}\n")
        f.write(f"Success: {suc_num}/{test_num}\n")
    print(f"Result saved to: {result_file}")


if __name__ == "__main__":
    main()
