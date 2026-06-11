#!/usr/bin/env python3
"""
测试二元奖励分类器的实时性能。

Live test for robosuite_door reward classifier.

This script:
1. opens robosuite viewer;
2. opens wrist / side camera preview;
3. lets you control the robot with robosuite Keyboard;
4. prints and overlays classifier probability in real time.

Keys:
  Backspace : reset episode
  q / Esc   : quit

启动命令:  #CPU测试，仅显示两个相机视角
    CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cpu python -B test_classifier_live.py

"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = SCRIPT_DIR.parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from experiments.robosuite_door.config import TrainConfig
from experiments.robosuite_door.env import unwrap_robosuite_door_env
from serl_launcher.networks.reward_classifier import load_classifier_func


HAS_RENDERER = False
RENDER_CAMERA = "mobilebase0_base_sideview"
MAX_EPISODE_STEPS = 1000
PREVIEW_SCALE = 3
CONTROL_HZ = 20
POS_SENSITIVITY = 0.30
ROT_SENSITIVITY = 0.30
THRESHOLD = 0.7


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def squeeze_rgb(obs, key: str) -> np.ndarray:
    image = np.asarray(obs[key])
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    assert image.ndim == 3 and image.shape[-1] == 3, f"{key} image shape: {image.shape}"
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def draw_text(frame, lines):
    import cv2

    y = 18
    for line in lines:
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18


def show_preview(obs, prob: float, logit: float, step: int, episode: int) -> int:
    import cv2

    wrist = squeeze_rgb(obs, "wrist")
    side = squeeze_rgb(obs, "side")
    frame = cv2.cvtColor(np.concatenate([wrist, side], axis=1), cv2.COLOR_RGB2BGR)

    if PREVIEW_SCALE != 1:
        frame = cv2.resize(frame, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE, interpolation=cv2.INTER_NEAREST)

    reward = int(prob > THRESHOLD)
    draw_text(
        frame,
        [
            f"episode {episode}  step {step}",
            f"classifier prob {prob:.4f}  logit {logit:.3f}  reward {reward}  threshold {THRESHOLD}",
            "expect: closed/grasped/partial-open -> 0, clearly-open -> 1",
            "Backspace reset | Esc/q quit",
        ],
    )

    cv2.imshow("robosuite_door_classifier_live_test", frame)
    return cv2.waitKey(1)


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
            raise RuntimeError(f"hotkeys unavailable: {exc}") from exc

    def on_press(self, key):
        char = getattr(key, "char", None)
        if char == "q":
            self.quit = True
        elif self.Key is not None and key == self.Key.esc:
            self.quit = True
        elif self.Key is not None and key == self.Key.backspace:
            self.reset = True

    def update_from_cv_key(self, key: int):
        if key < 0:
            return
        key = key & 0xFF
        if key in (27, ord("q")):
            self.quit = True
        elif key in (8, 127):
            self.reset = True

    def pop(self):
        reset, quit_now = self.reset, self.quit
        self.reset, self.quit = False, False
        return reset, quit_now

    def close(self):
        if self.listener is not None:
            self.listener.stop()


def compute_prob(classifier_func, obs) -> tuple[float, float]:
    import jax

    logits = classifier_func(obs)
    logits = jax.device_get(logits)
    logit = float(np.asarray(logits).reshape(-1)[0])
    prob = sigmoid(logit)
    return prob, logit


def reset_episode(env, device):
    obs, _ = env.reset()
    device.start_control()
    return obs


def main():
    import cv2
    import jax

    config = TrainConfig()

    env = config.get_environment(
        fake_env=False,
        save_video=False,
        classifier=False,
        has_renderer=HAS_RENDERER,
        render_camera=RENDER_CAMERA,
        max_episode_steps=MAX_EPISODE_STEPS,
    )

    door_env = unwrap_robosuite_door_env(env)
    rs_env = door_env.get_robosuite_env()
    device = make_device(rs_env)
    hotkeys = Hotkeys()

    ckpt_dir = SCRIPT_DIR / "classifier_ckpt"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"classifier checkpoint dir not found: {ckpt_dir}")

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", jax.devices())
    print("checkpoint dir:", ckpt_dir)

    classifier_func = load_classifier_func(
        key=jax.random.PRNGKey(0),
        sample=env.observation_space.sample(),
        image_keys=config.classifier_keys,
        checkpoint_path=str(ckpt_dir),
    )

    cv2.namedWindow("robosuite_door_classifier_live_test", cv2.WINDOW_NORMAL)

    episode = 1
    step = 0
    pending_reset = False
    obs = reset_episode(env, device)

    print("warming up classifier...")
    prob, logit = compute_prob(classifier_func, obs)
    print(f"initial prob={prob:.4f}, logit={logit:.3f}")

    try:
        while True:
            if pending_reset:
                episode += 1
                step = 0
                obs = reset_episode(env, device)
                pending_reset = False

            frame_start = time.time()

            prob, logit = compute_prob(classifier_func, obs)
            cv_key = show_preview(obs, prob, logit, step, episode)
            hotkeys.update_from_cv_key(cv_key)

            reset, quit_now = hotkeys.pop()
            if quit_now:
                break
            if reset:
                pending_reset = True
                continue

            action = device_action(device)
            if action is None:
                pending_reset = True
                continue

            obs, _, done, truncated, _ = env.step(action)
            if HAS_RENDERER:
                rs_env.render()

            step += 1
            if done or truncated or step >= MAX_EPISODE_STEPS:
                pending_reset = True

            sleep_time = max(0.0, 1.0 / CONTROL_HZ - (time.time() - frame_start))
            time.sleep(sleep_time)

    finally:
        hotkeys.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        env.close()


if __name__ == "__main__":
    main()
PY