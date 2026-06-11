#!/usr/bin/env python3

from pathlib import Path
import sys
import importlib
import inspect

THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
REPO_ROOT = THIS_FILE.parents[3]

sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from env import RobosuiteDoorHILSERLEnv


def make_config():
    for module_name in ("config", "env"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        for name in (
            "RobosuiteDoorConfig",
            "RobosuiteDoorEnvConfig",
            "DoorEnvConfig",
            "EnvConfig",
        ):
            cls = getattr(module, name, None)
            if cls is not None:
                return cls()

        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and "Config" in name:
                try:
                    return obj()
                except Exception:
                    pass

    return None


def make_env():
    config = make_config()
    if config is None:
        return RobosuiteDoorHILSERLEnv()
    return RobosuiteDoorHILSERLEnv(config=config)


def get_images_and_state(obs):
    if "images" in obs:
        wrist = obs["images"]["wrist"]
        side = obs["images"]["side"]
        state = obs["state"]
    else:
        wrist = obs["wrist"]
        side = obs["side"]
        state = obs["state"]
    return wrist, side, state


def main():
    out = Path("/tmp/robosuite_door_check")
    out.mkdir(parents=True, exist_ok=True)

    env = make_env()
    obs, info = env.reset()

    wrist, side, state = get_images_and_state(obs)

    cv2.imwrite(str(out / "wrist_000.png"), cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "side_000.png"), cv2.cvtColor(side, cv2.COLOR_RGB2BGR))
    np.save(out / "state_raw.npy", state, allow_pickle=True)

    print("saved to:", out)
    print("obs keys:", sorted(obs.keys()))
    print("wrist:", wrist.shape, wrist.dtype, int(wrist.min()), int(wrist.max()))
    print("side:", side.shape, side.dtype, int(side.min()), int(side.max()))

    if isinstance(state, dict):
        print("state keys:", sorted(state.keys()))
        for k, v in state.items():
            arr = np.asarray(v)
            print(k, arr.shape, arr.dtype, arr.flatten()[:8])
    else:
        arr = np.asarray(state)
        print("state:", arr.shape, arr.dtype, arr.flatten()[:20])

    env.close()


if __name__ == "__main__":
    main()
