from __future__ import annotations

import argparse
import numpy as np
from typing import Callable, Any

try:
    from .websocket_client_policy import WebsocketClientPolicy
except ImportError:
    from websocket_client_policy import WebsocketClientPolicy


def format_obs(obs: dict, prompt: str) -> dict:
    return {
        "observation.images.cam_high": obs["observation"]["head_camera"]["rgb"],
        "observation.images.cam_left_wrist": obs["observation"]["left_camera"]["rgb"],
        "observation.images.cam_right_wrist": obs["observation"]["right_camera"]["rgb"],
        "observation.state": obs["joint_action"]["vector"],
        "task": prompt,
    }


def build_dummy_obs(h: int = 480, w: int = 640) -> dict:
    return {
        "observation.images.cam_high": np.random.randint(0, 255, (h, w, 3), dtype=np.uint8),
        "observation.images.cam_left_wrist": np.random.randint(0, 255, (h, w, 3), dtype=np.uint8),
        "observation.images.cam_right_wrist": np.random.randint(0, 255, (h, w, 3), dtype=np.uint8),
        "observation.state": np.random.randn(16).astype(np.float32),
    }


def parse_action_steps(action: np.ndarray) -> list[np.ndarray]:
    action = np.asarray(action)
    if action.ndim == 2 and action.shape[1] in (14, 16):
        # New server output: [N, action_dim], typically [64,16]
        return [action[i].astype(np.float32) for i in range(action.shape[0])]
    if action.ndim == 3:
        # Backward-compatible old format: [action_dim, F, A]
        out = []
        for i in range(action.shape[1]):
            for j in range(action.shape[2]):
                out.append(action[:, i, j].astype(np.float32))
        return out
    raise ValueError(f"Unexpected action shape: {action.shape}")


def run_one_chunk_with_env(
    client: WebsocketClientPolicy,
    task_env: Any,
    prompt: str,
    format_obs_fn: Callable[[Any, str], dict],
    to_env_action_fn: Callable[[np.ndarray], np.ndarray],
    action_type: str = "ee",
) -> dict:
    """
    One full closed-loop chunk:
    1) query server with current obs
    2) execute all returned action steps in env
    3) keep 1 keyframe every 2 executed steps
    4) send keyframe sequence back to server via compute_kv_cache
    """
    first_obs = format_obs_fn(task_env.get_obs(), prompt)
    ret = client.infer({"obs": first_obs, "prompt": prompt})
    action_steps = parse_action_steps(ret["action"])

    key_frame_list = []
    for step_i, step_action in enumerate(action_steps):
        env_action = to_env_action_fn(step_action)
        task_env.take_action(env_action, action_type=action_type)
        if (step_i + 1) % 2 == 0:
            obs_i = format_obs_fn(task_env.get_obs(), prompt)
            key_frame_list.append(obs_i)

    client.infer({"obs": key_frame_list, "compute_kv_cache": True})
    return {"num_actions": len(action_steps), "num_keyframes": len(key_frame_list)}


def run_policy_once_with_dummy(client: WebsocketClientPolicy, prompt: str, total_steps: int = 64) -> None:
    first_obs = build_dummy_obs()
    ret = client.infer({"obs": first_obs, "prompt": prompt})
    action_steps = parse_action_steps(ret["action"])
    print("server action steps:", len(action_steps), "single action dim:", action_steps[0].shape[0])

    # Simulate execution, and save one observation every 2 executed actions.
    key_frame_list = []
    for step_i in range(min(total_steps, len(action_steps))):
        if (step_i + 1) % 2 == 0:
            key_frame_list.append(build_dummy_obs())

    client.infer({"obs": key_frame_list, "compute_kv_cache": True})
    print("compute_kv_cache frames sent:", len(key_frame_list))


def main():
    parser = argparse.ArgumentParser("RobotWin websocket client smoke test")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", type=str, default="pick and place")
    parser.add_argument("--dummy", action="store_true", help="Use random dummy observation for quick protocol test.")
    parser.add_argument("--steps", type=int, default=64, help="Number of action steps to simulate execution.")
    args = parser.parse_args()

    client = WebsocketClientPolicy(host=args.host, port=args.port)
    client.infer({"reset": True, "prompt": args.prompt})

    if args.dummy:
        run_policy_once_with_dummy(client, args.prompt, total_steps=args.steps)
        print("dummy protocol test finished")
        return

    raise RuntimeError(
        "This script is a websocket test client. "
        "For real RobotWin rollout, import parse_action_steps() and follow the same 'execute -> every2frame -> compute_kv_cache' flow."
    )


if __name__ == "__main__":
    main()
