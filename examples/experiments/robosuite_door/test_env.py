import argparse
import sys
from pathlib import Path

import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parents[2]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from experiments.robosuite_door.env import RobosuiteDoorEnvConfig, RobosuiteDoorHILSERLEnv


FORBIDDEN_OBS_KEYS = {
    "door_hinge_qpos",
    "hinge_qpos",
    "handle_qpos",
    "handle_pose",
    "door_pose",
    "success",
    "true_contact_point",
    "true_contact_force",
    "handle_to_eef",
    "handle_to_eef_pos",
    "door_to_eef",
    "door_to_eef_pos",
}


def quat_to_yaw_wxyz(quat):
    w, x, y, z = quat
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def door_xy_yaw(env):
    door_body_id = env.robosuite_env.sim.model.body_name2id(env.robosuite_env.door.root_body)
    pos = np.asarray(env.robosuite_env.sim.model.body_pos[door_body_id]).copy()
    quat = np.asarray(env.robosuite_env.sim.model.body_quat[door_body_id]).copy()
    return np.array([pos[0], pos[1], quat_to_yaw_wxyz(quat)])


def latch_z(env):
    latch_body_id = env.robosuite_env.object_body_ids["latch"]
    return float(env.robosuite_env.sim.model.body_pos[latch_body_id][2])


def base_qpos(env):
    state = env._ready_state["base_state"]
    return np.array(
        [
            env.robosuite_env.sim.data.qpos[
                env.robosuite_env.sim.model.get_joint_qpos_addr(state["base_joint_names"][name])
            ]
            for name in ("forward", "side", "yaw")
        ],
        dtype=np.float64,
    )


def arm_qpos(env):
    robot = env.robosuite_env.robots[0]
    return np.asarray(env.robosuite_env.sim.data.qpos[robot._ref_arm_joint_pos_indexes], dtype=np.float64).copy()


def assert_policy_obs_is_clean(obs):
    obs_keys = set(obs["images"].keys()) | set(obs["state"].keys())
    leaked = obs_keys & FORBIDDEN_OBS_KEYS
    assert not leaked, f"policy observation leaked simulator truth keys: {sorted(leaked)}"


def assert_obs_shapes(obs):
    assert set(obs["images"]) == {"wrist", "side"}
    for key, image in obs["images"].items():
        assert image.shape == (128, 128, 3), (key, image.shape)
        assert image.dtype == np.uint8, (key, image.dtype)
        assert image.min() >= 0 and image.max() <= 255, key

    assert obs["state"]["tcp_force"].shape == (3,)
    assert obs["state"]["tcp_torque"].shape == (3,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resets", type=int, default=5)
    parser.add_argument("--zero-steps", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    config = RobosuiteDoorEnvConfig(has_renderer=args.render, has_offscreen_renderer=True)
    env = RobosuiteDoorHILSERLEnv(config=config)
    try:
        assert env.action_space.shape == (7,)

        obs, info = env.reset()
        assert info == {}
        assert_policy_obs_is_clean(obs)
        assert_obs_shapes(obs)

        base_values = []
        arm_values = []
        door_values = []
        latch_values = []
        for _ in range(args.resets):
            obs, _ = env.reset()
            base_values.append(base_qpos(env))
            arm_values.append(arm_qpos(env))
            door_values.append(door_xy_yaw(env))
            latch_values.append(latch_z(env))
            assert_policy_obs_is_clean(obs)
            assert_obs_shapes(obs)

        for value in base_values[1:]:
            np.testing.assert_allclose(value, base_values[0], atol=1e-8)
        for value in arm_values[1:]:
            np.testing.assert_allclose(value, arm_values[0], atol=1e-8)

        door_values = np.asarray(door_values)
        latch_values = np.asarray(latch_values)
        assert np.max(np.ptp(door_values, axis=0)) > 1e-6, "Door x / y / yaw did not vary across resets"
        assert np.ptp(latch_values) > 1e-6, "latch / handle z did not vary across resets"
        assert np.all(np.abs(latch_values - np.mean(latch_values)) < 0.05), "latch z appears to be drifting"

        obs, _ = env.reset()
        start_base = base_qpos(env)
        start_arm = arm_qpos(env)
        for _ in range(args.zero_steps):
            obs, reward, done, truncated, info = env.step(np.zeros(7, dtype=np.float32))
            assert isinstance(reward, float)
            assert done is False
            assert isinstance(truncated, bool)
            assert info == {}
            assert_policy_obs_is_clean(obs)
        np.testing.assert_allclose(base_qpos(env), start_base, atol=1e-6)
        np.testing.assert_allclose(arm_qpos(env), start_arm, atol=5e-3)

        print("robosuite_door env checks passed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
