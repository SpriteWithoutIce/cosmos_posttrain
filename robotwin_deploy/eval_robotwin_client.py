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
    from .websocket_client_policy import WebsocketClientPolicy
    from .robotwin_client import parse_action_steps
except ImportError:
    from websocket_client_policy import WebsocketClientPolicy
    from robotwin_client import parse_action_steps


def ensure_robowin_importable(robowin_root: str) -> None:
    root = Path(robowin_root).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)


def write_json(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def class_decorator(task_name: str):
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def format_obs(observation: dict, prompt: str) -> dict:
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


def eval_policy(
    task_env: Any,
    model: WebsocketClientPolicy,
    test_num: int,
    save_root: str,
    task_name: str,
    seed: int = 0,
) -> tuple[int, int]:
    from envs.utils.create_actor import UnStableError

    task_env.suc = 0
    task_env.test_num = 0
    st_seed = 10000 * (1 + seed)
    now_seed = st_seed
    succ_seed = 0

    while succ_seed < test_num:
        try:
            task_env.setup_demo(now_ep_num=task_env.test_num, seed=now_seed, is_test=True)
            episode_info = task_env.play_once()
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
        task_env.setup_demo(now_ep_num=task_env.test_num, seed=now_seed, is_test=True)
        prompt = task_env.get_instruction()
        model.infer({"reset": True, "prompt": prompt})

        init_obs = task_env.get_obs()
        init_eef_pose = init_obs["endpose"]["left_endpose"] + [init_obs["endpose"]["left_gripper"]]
        init_eef_pose += init_obs["endpose"]["right_endpose"] + [init_obs["endpose"]["right_gripper"]]
        init_eef_pose = np.array(init_eef_pose, dtype=np.float64)

        while task_env.take_action_cnt < task_env.step_lim:
            first_obs = format_obs(task_env.get_obs(), prompt)
            ret = model.infer({"obs": first_obs, "prompt": prompt})
            action_steps = parse_action_steps(ret["action"])
            key_frame_list = []

            for step_i, raw_step in enumerate(action_steps):
                ee_action = to_env_ee_action(np.asarray(raw_step), init_eef_pose)
                task_env.take_action(ee_action, action_type="ee")
                if (step_i + 1) % 2 == 0:
                    key_frame_list.append(format_obs(task_env.get_obs(), prompt))
                if task_env.eval_success:
                    break

            model.infer({"obs": key_frame_list, "compute_kv_cache": True})
            if task_env.eval_success:
                break

        succ = bool(task_env.eval_success)
        if succ:
            task_env.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        task_env.close_env(clear_cache=True)
        task_env.test_num += 1
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

    return st_seed, task_env.suc


def main() -> None:
    parser = argparse.ArgumentParser("RobotWin eval client for cosmos_posttrain server")
    parser.add_argument("--robowin_root", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--test_num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_root", type=str, default="./results")
    parser.add_argument("--extra_config", type=str, default="", help="Optional yaml config to merge into args.")
    args = parser.parse_args()

    if args.extra_config:
        with open(args.extra_config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    ensure_robowin_importable(args.robowin_root)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    task_env = class_decorator(args.task_name)
    model = WebsocketClientPolicy(host=args.host, port=args.port)
    st_seed, suc_num = eval_policy(
        task_env=task_env,
        model=model,
        test_num=args.test_num,
        save_root=args.save_root,
        task_name=args.task_name,
        seed=args.seed,
    )

    result_file = Path(args.save_root) / f"stseed-{st_seed}" / "metrics" / args.task_name / "_result.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"Timestamp: {current_time}\n")
        f.write(f"Task: {args.task_name}\n")
        f.write(f"Success: {suc_num}/{args.test_num}\n")
    print(f"Result saved to: {result_file}")


if __name__ == "__main__":
    main()

