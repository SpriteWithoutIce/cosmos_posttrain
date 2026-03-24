"""RoboTwin evaluation client for Cosmos RoboTwin Policy.

Connects to the inference server via websocket, runs RoboTwin episodes,
and records success metrics.

Usage (from any directory):
    python eval_cosmos_client.py --config <task_yml> [--overrides ...]
"""

import sys
import os
import subprocess
import functools
import time
import logging
from pathlib import Path

import cv2
import numpy as np
import imageio
import yaml
import json
import argparse
import importlib
import traceback
from datetime import datetime
from scipy.spatial.transform import Rotation as R

import msgpack
import websockets.sync.client

# ---------------------------------------------------------------------------
# RoboTwin bootstrap
# ---------------------------------------------------------------------------
ROBOTWIN_ROOT = Path(os.environ.get("ROBOTWIN_ROOT", "/root/linyihan/RoboTwin"))
if str(ROBOTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT))
os.chdir(ROBOTWIN_ROOT)

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
from description.utils.generate_episode_instructions import generate_episode_descriptions

# ---------------------------------------------------------------------------
# Inline msgpack-numpy helpers
# ---------------------------------------------------------------------------

def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(),
                b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]),
                          shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_Packer = functools.partial(msgpack.Packer, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


# ---------------------------------------------------------------------------
# Websocket client
# ---------------------------------------------------------------------------

class WebsocketClientPolicy:
    def __init__(self, host: str = "127.0.0.1", port: int = 29055):
        self._uri = f"ws://{host}:{port}"
        self._packer = _Packer()
        self._ws, self._meta = self._wait_for_server()

    def _wait_for_server(self):
        while True:
            try:
                print(f"[Client] Connecting to {self._uri} …", flush=True)
                conn = websockets.sync.client.connect(
                    self._uri, open_timeout=15, close_timeout=10,
                    compression=None, max_size=None, ping_interval=None,
                )
                meta = _unpackb(conn.recv())
                print("[Client] Connected.", flush=True)
                return conn, meta
            except Exception as e:
                print(f"[Client] Not ready: {e}. Retry in 5 s …", flush=True)
                time.sleep(5)

    def infer(self, obs: dict) -> dict:
        self._ws.send(self._packer.pack(obs))
        resp = self._ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"Server error:\n{resp}")
        return _unpackb(resp)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_obs(observation, prompt):
    state = np.array(
        observation["endpose"]["left_endpose"]
        + [observation["endpose"]["left_gripper"]]
        + observation["endpose"]["right_endpose"]
        + [observation["endpose"]["right_gripper"]],
        dtype=np.float32,
    )
    return {
        "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"],
        "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
        "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],
        "observation.state": state,
        "task": prompt,
    }


def add_eef_pose(new_pose, init_pose):
    new_R = R.from_quat(new_pose[3:7][None])
    init_R = R.from_quat(init_pose[3:7][None])
    out_rot = (init_R * new_R).as_quat().reshape(-1)
    out_trans = new_pose[:3] + init_pose[:3]
    return np.concatenate([out_trans, out_rot, new_pose[7:8]])


def add_init_pose(new_pose, init_pose):
    left = add_eef_pose(new_pose[:8], init_pose[:8])
    right = add_eef_pose(new_pose[8:], init_pose[8:])
    return np.concatenate([left, right])


def write_json(data: dict, fpath: Path):
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def class_decorator(task_name):
    mod = importlib.import_module(f"envs.{task_name}")
    try:
        return getattr(mod, task_name)()
    except Exception:
        raise SystemExit("No Task")


def get_camera_config(camera_type):
    cfg_path = os.path.join(ROBOTWIN_ROOT, "task_config/_camera_config.yml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    return args[camera_type]


def get_embodiment_config(robot_file):
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def add_title_bar(img, text, font_scale=0.8, thickness=2):
    h, w, _ = img.shape
    bar_h = 40
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.putText(bar, text, ((w - tw) // 2, (bar_h + th) // 2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return np.vstack([bar, img])


def save_comparison_video(real_obs_list, save_path, fps=15):
    if not real_obs_list:
        return
    frames = []
    for obs in real_obs_list:
        cam_h = obs["observation.images.cam_high"]
        cam_l = obs["observation.images.cam_left_wrist"]
        cam_r = obs["observation.images.cam_right_wrist"]
        base_h = cam_h.shape[0]

        def _resize(img, h):
            if img.shape[0] != h:
                w = int(img.shape[1] * h / img.shape[0])
                img = cv2.resize(img, (w, h))
            img = np.ascontiguousarray(img)
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            return img

        row = np.hstack([_resize(cam_h, base_h), _resize(cam_l, base_h), _resize(cam_r, base_h)])
        row = add_title_bar(np.ascontiguousarray(row), "Real Observation (High / Left / Right)")
        frames.append(row)
    imageio.mimsave(save_path, frames, fps=fps)
    print(f"Video saved: {save_path}")


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def eval_policy(
    task_name, TASK_ENV, args, model,
    st_seed, test_num=100, video_size=None,
    instruction_type="seen",
):
    print(f"\033[34mTask: {task_name}  Policy: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0
    now_id = 0
    succ_seed = 0
    now_seed = st_seed
    clear_cache_freq = args["clear_cache_freq"]
    args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError:
                TASK_ENV.close_env(); now_seed += 1; args["render_freq"] = render_freq; continue
            except Exception as e:
                TASK_ENV.close_env(); now_seed += 1; args["render_freq"] = render_freq
                print(f"Error: {e}"); traceback.print_exc(); continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
        else:
            now_seed += 1; args["render_freq"] = render_freq; continue

        args["render_freq"] = render_freq
        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(task_name, episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                 "-pixel_format", "rgb24", "-video_size", video_size,
                 "-framerate", "10", "-i", "-", "-pix_fmt", "yuv420p",
                 "-vcodec", "libx264", "-crf", "23",
                 f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4"],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        prompt = TASK_ENV.get_instruction()
        model.infer({"reset": True, "prompt": prompt, "task_name": task_name})

        full_obs_list = []
        initial_obs = TASK_ENV.get_obs()
        full_obs_list.append(format_obs(initial_obs, prompt))

        # ---- main action loop ----
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            obs_dict = format_obs(observation, prompt)

            ret = model.infer({"obs": obs_dict, "prompt": prompt})
            action = ret["action"]  # (action_dim, F_half, N)
            # print(action.shape)

            for i in range(action.shape[1]):
                for j in range(action.shape[2]):
                    if TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
                        break

                    ee_action = action[:, i, j].copy()

                    if action.shape[0] == 16:
                        # ee_action = add_init_pose(ee_action, init_eef)
                        ee_action = np.concatenate([
                            ee_action[:3],
                            ee_action[3:7] / np.linalg.norm(ee_action[3:7]),
                            ee_action[7:11],
                            ee_action[11:15] / np.linalg.norm(ee_action[11:15]),
                            ee_action[15:16],
                        ])
                    elif action.shape[0] == 14:
                        from evaluation.robotwin.geometry import euler2quat
                        ee_action = np.concatenate([
                            ee_action[:3],
                            euler2quat(ee_action[3], ee_action[4], ee_action[5]),
                            ee_action[6:10],
                            euler2quat(ee_action[10], ee_action[11], ee_action[12]),
                            ee_action[13:14],
                        ])
                    else:
                        raise NotImplementedError(f"action_dim={action.shape[0]} not supported")

                    TASK_ENV.take_action(ee_action, action_type="ee")

                    full_obs_list.append(format_obs(TASK_ENV.get_obs(), prompt))

                if TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
                    break

            if TASK_ENV.eval_success:
                succ = True
                break

        # ---- save visualisation ----
        vis_dir = Path(args["save_root"]) / f"stseed-{st_seed}" / "visualization" / task_name
        vis_dir.mkdir(parents=True, exist_ok=True)
        video_name = f"{TASK_ENV.test_num}_{prompt.replace(' ', '_')}_{succ}.mp4"
        save_comparison_video(full_obs_list, str(vis_dir / video_name))

        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()
        TASK_ENV.test_num += 1

        # persist metrics
        metrics_dir = Path(args["save_root"]) / f"stseed-{st_seed}" / "metrics" / task_name
        metrics_dir.mkdir(parents=True, exist_ok=True)
        write_json({
            "succ_num": float(TASK_ENV.suc),
            "total_num": float(TASK_ENV.test_num),
            "succ_rate": float(TASK_ENV.suc / TASK_ENV.test_num),
        }, metrics_dir / "res.json")

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | "
            f"\033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m "
            f"=> \033[95m{round(TASK_ENV.suc / TASK_ENV.test_num * 100, 1)}%\033[0m  "
            f"seed: \033[90m{now_seed}\033[0m\n"
        )
        now_seed += 1

    return now_seed, TASK_ENV.suc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_single_task(task_name, model, usr_args, current_time):
    """Run evaluation for one task using an already-connected model client."""
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    save_root = usr_args["save_root"]
    policy_name = usr_args["policy_name"]

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    args["save_root"] = save_root

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f, Loader=yaml.FullLoader)

    def _get_embodiment_file(etype):
        return _embodiment_types[etype]["file_path"]

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f, Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = _get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = _get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = _get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = _get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    video_size = None
    if args["eval_video_log"]:
        cam_cfg = get_camera_config(head_camera_type)
        video_size = f"{cam_cfg['w']}x{cam_cfg['h']}"
        args["eval_video_save_dir"] = save_dir

    args["policy_name"] = policy_name

    TASK_ENV = class_decorator(task_name)
    seed = usr_args["seed"]
    test_num = usr_args["test_num"]
    st_seed = 10000 * (1 + seed)

    st_seed, suc_num = eval_policy(
        task_name, TASK_ENV, args, model,
        st_seed, test_num=test_num,
        video_size=video_size,
    )

    result_path = os.path.join(save_dir, "_result.txt")
    with open(result_path, "w") as f:
        f.write(f"Timestamp: {current_time}\n")
        f.write(f"Success: {suc_num}/{test_num}\n")
    print(f"Results saved to {result_path}")
    return suc_num


ALL_TASK_GROUPS = [
    "adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size click_alarmclock dump_bin_bigbin grab_roller handover_block handover_mic hanging_mug lift_pot move_can_pot move_pillbottle_pad move_playingcard_away move_stapler_pad open_laptop open_microwave pick_dual_bottles pick_diverse_bottles",
    # "adjust_bottle place_mouse_pad dump_bin_bigbin move_pillbottle_pad pick_dual_bottles shake_bottle place_fan turn_switch",
    # "stack_bowls_three handover_block hanging_mug scan_object lift_pot put_object_cabinet stack_blocks_three place_shoe",
    # "shake_bottle_horizontally place_container_plate rotate_qrcode place_object_stand put_bottles_dustbin move_stapler_pad place_burger_fries place_bread_basket",
    # "pick_diverse_bottles open_microwave beat_block_hammer press_stapler click_bell move_playingcard_away open_laptop move_can_pot",
    # "stack_bowls_two place_a2b_right stamp_seal place_object_basket handover_mic place_bread_skillet stack_blocks_two place_cans_plasticbox",
    # "click_alarmclock blocks_ranking_size place_phone_stand place_can_basket place_object_scale place_a2b_left grab_roller place_dual_shoes",
    # "place_empty_cup blocks_ranking_rgb",
]


def resolve_task_list(task_spec: str) -> list[str]:
    """Resolve a task specifier to a list of task names.

    task_spec can be:
      - "all"           → every task across all groups
      - "0" .. "6"      → all tasks in that group
      - "adjust_bottle"  → single task
      - "adjust_bottle,place_fan" → comma-separated list
    """
    if task_spec == "all":
        tasks = []
        for g in ALL_TASK_GROUPS:
            tasks.extend(g.split())
        return tasks
    if task_spec.isdigit() and 0 <= int(task_spec) < len(ALL_TASK_GROUPS):
        return ALL_TASK_GROUPS[int(task_spec)].split()
    if "," in task_spec:
        return [t.strip() for t in task_spec.split(",") if t.strip()]
    return [task_spec]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Task YAML config (e.g. policy/ACT/deploy_policy.yml)")
    parser.add_argument("--tasks", type=str, default="adjust_bottle",
                        help="Task specifier: task name, comma-list, group id (0-6), or 'all'")
    parser.add_argument("--port", type=int, default=29055)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--save_root", type=str, default="results/cosmos_eval")
    parser.add_argument("--test_num", type=int, default=100)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.overrides:
        for i in range(0, len(args.overrides), 2):
            key = args.overrides[i].lstrip("--")
            val = args.overrides[i + 1]
            try:
                val = eval(val)
            except Exception:
                pass
            config[key] = val

    config.setdefault("save_root", args.save_root)
    config.setdefault("test_num", args.test_num)
    config.setdefault("host", args.host)
    config.setdefault("port", args.port)

    task_list = resolve_task_list(args.tasks)
    print(f"[Client] Will evaluate {len(task_list)} task(s): {task_list}")

    # connect to server ONCE
    model = WebsocketClientPolicy(host=args.host, port=args.port)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary: dict[str, str] = {}

    for idx, task_name in enumerate(task_list):
        print(f"\n\033[33m========== [{idx+1}/{len(task_list)}] {task_name} ==========\033[0m")
        config["task_name"] = task_name
        try:
            suc = run_single_task(task_name, model, config, current_time)
            summary[task_name] = f"{suc}/{config['test_num']}"
        except Exception as e:
            print(f"\033[91mTask {task_name} failed: {e}\033[0m")
            traceback.print_exc()
            summary[task_name] = "ERROR"

    # print final summary
    print("\n\033[36m" + "=" * 50)
    print("  EVALUATION SUMMARY")
    print("=" * 50 + "\033[0m")
    for t, r in summary.items():
        print(f"  {t:40s} {r}")
    print()


if __name__ == "__main__":
    from robotwin_deploy.test_render import Sapien_TEST
    Sapien_TEST()
    main()
