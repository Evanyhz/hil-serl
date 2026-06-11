#!/usr/bin/env python3
"""Check Door handle height randomization over repeated resets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = SCRIPT_DIR.parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from experiments.robosuite_door.config import TrainConfig
from experiments.robosuite_door.env import unwrap_robosuite_door_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resets", type=int, default=100)
    parser.add_argument("--handle-z-randomization", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()
    env = TrainConfig().get_environment(
        fake_env=False,
        save_video=False,
        classifier=False,
        has_renderer=False,
        handle_z_randomization=args.handle_z_randomization,
    )
    door_env = unwrap_robosuite_door_env(env)
    rs_env = door_env.get_robosuite_env()

    handle_zs = []
    latch_local_zs = []
    try:
        for _ in range(args.resets):
            obs, _ = env.reset()
            assert set(obs.keys()) == {"wrist", "side", "state"}
            rs_env.sim.forward()
            handle_zs.append(float(rs_env.sim.data.site_xpos[rs_env.door_handle_site_id][2]))
            latch_local_zs.append(float(rs_env.sim.model.body_pos[rs_env.object_body_ids["latch"]][2]))

        handle_zs = np.asarray(handle_zs)
        offsets = np.asarray(latch_local_zs) - float(rs_env._latch_body_nominal_pos[2])
        limit = abs(args.handle_z_randomization) + 1e-6

        assert offsets.min() < 0.0 < offsets.max(), "handle z did not sample both below and above nominal"
        assert np.max(np.abs(offsets)) <= limit, "latch z drifted outside nominal randomization range"
        assert np.ptp(handle_zs) <= 2.0 * abs(args.handle_z_randomization) + 2e-6, "handle z range is too large"

        print(f"resets: {args.resets}")
        print(f"handle z min/max/mean: {handle_zs.min():.6f} / {handle_zs.max():.6f} / {handle_zs.mean():.6f}")
        print(f"latch local z offset min/max/mean: {offsets.min():.6f} / {offsets.max():.6f} / {offsets.mean():.6f}")
        print("handle z randomization check passed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
