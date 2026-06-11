#!/usr/bin/env python3
"""Collect transition-level reward-classifier data for robosuite_door.

This collector is intentionally task-specific: it always opens the robosuite
viewer, always shows the wrist/side classifier preview, and always saves data to
this experiment folder's classifier_data directory.
"""

from __future__ import annotations

import copy
import datetime as dt
import pickle as pkl
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = SCRIPT_DIR.parents[1]
OUTPUT_DIR = SCRIPT_DIR / "classifier_data"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from experiments.robosuite_door.config import TrainConfig
from experiments.robosuite_door.env import unwrap_robosuite_door_env


SUCCESSES_NEEDED = 200
FAILURES_NEEDED = 400
SAVE_EVERY = 20
MAX_EPISODE_STEPS = 1000

HAS_RENDERER = True
RENDER_CAMERA = "mobilebase0_base_sideview"
PREVIEW_SCALE = 3
POS_SENSITIVITY = 0.30
ROT_SENSITIVITY = 0.30
CONTROL_HZ = 20

OBS_KEYS = {"wrist", "side", "state"}
TRANSITION_KEYS = {
    "observations",
    "actions",
    "next_observations",
    "rewards",
    "masks",
    "dones",
}
FORBIDDEN_KEYS = {
    "images",
    "hinge_qpos",
    "handle_qpos",
    "handle_pos",
    "door_pos",
    "door_pose",
    "handle_pose",
    "success",
    "succeed",
    "is_success",
    "true_contact_point",
    "true_contact_force",
    "handle_to_eef",
    "door_to_eef",
}


def assert_no_forbidden_keys(value: Any, where: str = "transition") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key = str(key)
            assert key not in FORBIDDEN_KEYS, f"forbidden key at {where}.{key}"
            assert_no_forbidden_keys(child, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_no_forbidden_keys(child, f"{where}[{index}]")


def sanitize_observation(obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    assert set(obs.keys()) == OBS_KEYS, f"obs keys must be {sorted(OBS_KEYS)}, got {sorted(obs.keys())}"
    out = {key: np.asarray(obs[key]).copy() for key in sorted(OBS_KEYS)}
    assert_no_forbidden_keys(out, "observation")
    return out


def sanitize_transition(transition: Mapping[str, Any]) -> dict[str, Any]:
    assert set(transition.keys()) == TRANSITION_KEYS, (
        f"transition keys must be {sorted(TRANSITION_KEYS)}, got {sorted(transition.keys())}"
    )
    out = {
        "observations": sanitize_observation(transition["observations"]),
        "actions": np.asarray(transition["actions"], dtype=np.float32).reshape(7).copy(),
        "next_observations": sanitize_observation(transition["next_observations"]),
        "rewards": float(transition["rewards"]),
        "masks": float(transition["masks"]),
        "dones": bool(transition["dones"]),
    }
    assert_no_forbidden_keys(out)
    return copy.deepcopy(out)


def make_transition(obs, action, next_obs, reward, done) -> dict[str, Any]:
    return sanitize_transition(
        {
            "observations": obs,
            "actions": action,
            "next_observations": next_obs,
            "rewards": reward,
            "masks": 1.0 - float(done),
            "dones": bool(done),
        }
    )


def make_current_obs_transition(obs, action=None) -> dict[str, Any]:
    if action is None:
        action = np.zeros(7, dtype=np.float32)
    return make_transition(obs, action, obs, reward=0.0, done=False)


def make_env():
    config = TrainConfig()
    env = config.get_environment(
        fake_env=False,
        save_video=False,
        classifier=False,
        has_renderer=HAS_RENDERER,
        render_camera=RENDER_CAMERA,
        max_episode_steps=MAX_EPISODE_STEPS,
    )
    assert set(env.observation_space.spaces.keys()) == OBS_KEYS
    assert env.action_space.shape == (7,), env.action_space.shape
    return env


def squeeze_rgb(obs: Mapping[str, Any], key: str) -> np.ndarray:
    image = np.asarray(obs[key])
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    assert image.ndim == 3 and image.shape[-1] == 3, f"{key} image shape: {image.shape}"
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def draw_text(frame, lines: list[str]):
    import cv2

    y = 18
    for line in lines:
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18


def show_preview(obs, stats, last_action: np.ndarray, last_label: str) -> int:
    import cv2

    wrist = squeeze_rgb(obs, "wrist")
    side = squeeze_rgb(obs, "side")
    frame = cv2.cvtColor(np.concatenate([wrist, side], axis=1), cv2.COLOR_RGB2BGR)
    if PREVIEW_SCALE != 1:
        frame = cv2.resize(frame, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE, interpolation=cv2.INTER_NEAREST)

    draw_text(
        frame,
        [
            f"episode {stats['episode']}  step {stats['step']}",
            f"success {stats['success']} / {SUCCESSES_NEEDED}   failure {stats['failure']} / {FAILURES_NEEDED}",
            "last_action " + np.array2string(last_action, precision=2, suppress_small=True),
            f"last_label {last_label}",
            "1 success | 0 failure | Backspace reset | Esc/q quit",
        ],
    )
    cv2.imshow("robosuite_door_classifier", frame)
    return cv2.waitKey(1)


def device_action(device) -> np.ndarray | None:
    ac_dict = device.input2action()
    if ac_dict is None:
        return None

    action = np.zeros(7, dtype=np.float32)
    delta = ac_dict.get("right_delta")
    if delta is None:
        delta = next((v for k, v in ac_dict.items() if k.endswith("_delta")), None)
    if delta is not None:
        action[:6] = np.asarray(delta, dtype=np.float32).reshape(-1)[:6]

    gripper = ac_dict.get("right_gripper")
    if gripper is None:
        gripper = next((v for k, v in ac_dict.items() if k.endswith("_gripper")), None)
    if gripper is not None:
        action[6] = float(np.clip(np.asarray(gripper, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))
    return np.clip(action, -1.0, 1.0)


class Hotkeys:
    def __init__(self):
        self.label = None
        self.reset = False
        self.quit = False
        self.listener = None
        self.Key = None
        try:
            from pynput import keyboard

            self.Key = keyboard.Key
            self.listener = keyboard.Listener(on_press=self.on_press)
            self.listener.start()
        except Exception as exc:
            raise RuntimeError(f"label hotkeys unavailable: {exc}") from exc

    def on_press(self, key):
        char = getattr(key, "char", None)
        if char == "1":
            self.label = "success"
        elif char == "0":
            self.label = "failure"
        elif char == "q":
            self.quit = True
        elif self.Key is not None and key == self.Key.esc:
            self.quit = True
        elif self.Key is not None and key == self.Key.backspace:
            self.reset = True

    def update_from_cv_key(self, key: int) -> None:
        if key < 0:
            return
        key = key & 0xFF
        if key == ord("1"):
            self.label = "success"
        elif key == ord("0"):
            self.label = "failure"
        elif key in (27, ord("q")):
            self.quit = True
        elif key in (8, 127):
            self.reset = True

    def pop(self):
        label, reset, quit_now = self.label, self.reset, self.quit
        self.label, self.reset, self.quit = None, False, False
        return label, reset, quit_now

    def close(self) -> None:
        if self.listener is not None:
            self.listener.stop()


def save_buffers(successes, failures, stamp: str) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    success_path = OUTPUT_DIR / f"robosuite_door_success_images_{stamp}.pkl"
    failure_path = OUTPUT_DIR / f"robosuite_door_failure_images_{stamp}.pkl"
    for path, data in ((success_path, successes), (failure_path, failures)):
        with path.open("wb") as f:
            pkl.dump([sanitize_transition(t) for t in data], f)
    return success_path, failure_path


def make_device(rs_env):
    try:
        from robosuite.devices import Keyboard

        device = Keyboard(
            env=rs_env,
            pos_sensitivity=POS_SENSITIVITY,
            rot_sensitivity=ROT_SENSITIVITY,
        )
        viewer = getattr(rs_env, "viewer", None)
        if viewer is not None and hasattr(viewer, "add_keypress_callback"):
            viewer.add_keypress_callback(device.on_press)
        device.start_control()
        return device
    except Exception as exc:
        raise RuntimeError(f"robosuite Keyboard unavailable: {exc}") from exc


def print_counts(successes, failures):
    print(f"success transitions: {len(successes)} / {SUCCESSES_NEEDED}")
    print(f"failure transitions: {len(failures)} / {FAILURES_NEEDED}")


def targets_reached(successes, failures) -> bool:
    return len(successes) >= SUCCESSES_NEEDED and len(failures) >= FAILURES_NEEDED


def reset_episode(env, device, episode: int):
    obs, _ = env.reset()
    sanitize_observation(obs)
    device.start_control()
    return obs, episode + 1, 0


def main() -> None:
    env = make_env()
    door_env = unwrap_robosuite_door_env(env)
    rs_env = door_env.get_robosuite_env()
    device = make_device(rs_env)
    hotkeys = Hotkeys()

    successes, failures = [], []
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    obs, _ = env.reset()
    sanitize_observation(obs)
    print("obs keys:", sorted(obs.keys()))
    print("action shape:", env.action_space.shape)
    print("output dir:", OUTPUT_DIR)

    import cv2

    cv2.namedWindow("robosuite_door_classifier", cv2.WINDOW_NORMAL)

    episode, step = 1, 0
    last_action = np.zeros(7, dtype=np.float32)
    last_label = "none"
    pending_reset = False

    try:
        while not targets_reached(successes, failures):
            if pending_reset:
                obs, episode, step = reset_episode(env, device, episode)
                pending_reset = False
                continue

            frame_start = time.time()
            cv_key = show_preview(
                obs,
                {"episode": episode, "step": step, "success": len(successes), "failure": len(failures)},
                last_action,
                last_label,
            )
            hotkeys.update_from_cv_key(cv_key)

            label, reset, quit_now = hotkeys.pop()
            if label == "success":
                successes.append(make_current_obs_transition(obs, last_action))
                last_label = "success"
                print_counts(successes, failures)
            elif label == "failure":
                failures.append(make_current_obs_transition(obs, last_action))
                last_label = "failure"
                print_counts(successes, failures)

            total = len(successes) + len(failures)
            if label is not None and SAVE_EVERY > 0 and total > 0 and total % SAVE_EVERY == 0:
                paths = save_buffers(successes, failures, stamp)
                print(f"autosaved: {paths[0]}  {paths[1]}")

            if quit_now:
                break

            if reset:
                pending_reset = True
                continue

            action = device_action(device)
            if action is None:
                pending_reset = True
                continue

            next_obs, _, done, truncated, _ = env.step(action)
            rs_env.render()
            sanitize_observation(next_obs)

            last_action = action.copy()
            obs = next_obs
            step += 1
            pending_reset = bool(done or truncated or step >= MAX_EPISODE_STEPS)

            sleep_time = max(0.0, 1.0 / CONTROL_HZ - (time.time() - frame_start))
            time.sleep(sleep_time)
    finally:
        success_path, failure_path = save_buffers(successes, failures, stamp)
        print(f"saved: {success_path}")
        print(f"saved: {failure_path}")
        hotkeys.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        env.close()


if __name__ == "__main__":
    main()
