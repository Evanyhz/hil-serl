import copy
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np


MOBILE_CONTROLLER_CONFIG = {
    "type": "HYBRID_MOBILE_BASE",
    "body_parts": {
        "right": {
            "type": "OSC_POSE",
            "input_max": 1,
            "input_min": -1,
            "output_max": [0.01, 0.01, 0.01, 0.50, 0.50, 0.50],
            "output_min": [-0.01, -0.01, -0.01, -0.50, -0.50, -0.50],
            "kp": 150,
            "damping_ratio": 1,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300],
            "damping_ratio_limits": [0, 10],
            "position_limits": None,
            "orientation_limits": None,
            "uncouple_pos_ori": True,
            "input_type": "delta",
            "input_ref_frame": "base",
            "interpolation": None,
            "ramp_ratio": 0.2,
            "gripper": {
                "type": "GRIP",
            },
        },
        "base": {
            "type": "JOINT_VELOCITY",
            "interpolation": None,
        },
        "torso": {
            "type": "JOINT_POSITION",
            "interpolation": None,
            "kp": 2000,
        },
    },
}


@dataclass
class RobosuiteDoorEnvConfig:
    env_name: str = "Door"
    robot: str = "Fairino5V6"
    base: Optional[str] = "OmronMobileBase"
    gripper: str = "Robotiq85Gripper"
    mobile_robot_name: Optional[str] = None
    controller: Optional[str] = None
    env_configuration: str = "default"
    control_freq: int = 20
    max_episode_steps: int = 120
    reward_shaping: bool = False
    hard_reset: bool = False
    has_renderer: bool = False
    has_offscreen_renderer: bool = True
    use_camera_obs: bool = True
    use_object_obs: bool = False
    ignore_done: bool = True
    render_camera: str = "mobilebase0_base_sideview"
    render_gpu_device_id: int = -1
    camera_names: Tuple[str, str] = ("robot0_eye_in_hand", "mobilebase0_base_sideview")
    camera_heights: int = 128
    camera_widths: int = 128
    image_keys: Tuple[str, str] = ("wrist", "side")
    image_size: Tuple[int, int] = (128, 128)
    handle_z_randomization: float = 0.01
    mobile_base_door_distance: float = 0.90
    mobile_base_yaw: float = 0.5 * np.pi
    mobile_base_lateral_offset: float = 0.0
    handle_pregrasp_distance: float = 0.10
    handle_pregrasp_source: str = "geom"
    handle_pregrasp_orientation: str = "front"
    skip_handle_pregrasp_init: bool = False
    arm_qpos_dim: int = 6
    gripper_qpos_dim: int = 6
    action_low: Sequence[float] = field(default_factory=lambda: [-1.0] * 7)
    action_high: Sequence[float] = field(default_factory=lambda: [1.0] * 7)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        raise ValueError(f"Cannot normalize near-zero vector: {vec}")
    return vec / norm


def _get_unwrapped_env(env):
    return env.unwrapped if hasattr(env, "unwrapped") else env


def _get_mobile_base_refs(env) -> Optional[Dict[str, Any]]:
    robot = env.robots[0]
    base_model = robot.robot_model.base
    if base_model is None:
        return None

    sim = env.sim
    joint_names = {
        "forward": base_model.correct_naming("joint_mobile_forward"),
        "side": base_model.correct_naming("joint_mobile_side"),
        "yaw": base_model.correct_naming("joint_mobile_yaw"),
    }
    try:
        return {
            "base_model": base_model,
            "joint_names": joint_names,
            "qpos_addrs": {key: sim.model.get_joint_qpos_addr(name) for key, name in joint_names.items()},
            "qvel_addrs": {key: sim.model.get_joint_qvel_addr(name) for key, name in joint_names.items()},
            "base_body_id": sim.model.body_name2id(base_model.correct_naming("base")),
        }
    except Exception:
        return None


def _control_forward_dir_from_yaw(yaw: float) -> np.ndarray:
    return -np.array([np.cos(yaw), np.sin(yaw)])


def _place_mobile_base_in_front_of_table(
    env,
    table_distance: float,
    yaw: float,
    lateral_offset: float,
) -> Optional[Dict[str, Any]]:
    refs = _get_mobile_base_refs(env)
    if refs is None or not hasattr(env, "table_offset"):
        return None

    sim = env.sim
    qpos_addrs = refs["qpos_addrs"]
    qvel_addrs = refs["qvel_addrs"]
    base_body_id = refs["base_body_id"]

    table_center = np.asarray(env.table_offset[:2], dtype=float)
    current_base_xy = np.asarray(sim.data.body_xpos[base_body_id][:2], dtype=float)
    control_forward_dir = _control_forward_dir_from_yaw(yaw)
    lateral_dir = np.array([-control_forward_dir[1], control_forward_dir[0]])
    desired_base_xy = table_center - control_forward_dir * table_distance + lateral_dir * lateral_offset

    current_qpos_xy = np.array([sim.data.qpos[qpos_addrs["forward"]], sim.data.qpos[qpos_addrs["side"]]])
    zero_base_xy = current_base_xy - current_qpos_xy
    desired_qpos_xy = desired_base_xy - zero_base_xy

    sim.data.qpos[qpos_addrs["forward"]] = desired_qpos_xy[0]
    sim.data.qpos[qpos_addrs["side"]] = desired_qpos_xy[1]
    sim.data.qpos[qpos_addrs["yaw"]] = yaw
    sim.forward()

    for _ in range(2):
        actual_base_xy = np.asarray(sim.data.body_xpos[base_body_id][:2], dtype=float)
        base_error = desired_base_xy - actual_base_xy
        sim.data.qpos[qpos_addrs["forward"]] += base_error[0]
        sim.data.qpos[qpos_addrs["side"]] += base_error[1]
        sim.forward()

    for addr in qvel_addrs.values():
        sim.data.qvel[addr] = 0.0

    sim.forward()
    return {
        "base_qpos": {key: float(sim.data.qpos[qpos_addrs[key]]) for key in qpos_addrs},
        "base_joint_names": refs["joint_names"],
        "yaw": float(yaw),
        "table_center": table_center.copy(),
    }


def _set_mobile_base_qpos(env, base_state: Optional[Dict[str, Any]]) -> None:
    refs = _get_mobile_base_refs(env)
    if refs is None or base_state is None:
        return

    for key, qpos in base_state["base_qpos"].items():
        env.sim.data.qpos[refs["qpos_addrs"][key]] = qpos
    for addr in refs["qvel_addrs"].values():
        env.sim.data.qvel[addr] = 0.0
    env.sim.forward()


def _get_eef_pose(env, arm: str = "right") -> Tuple[np.ndarray, np.ndarray]:
    site_id = env.robots[0].eef_site_id[arm]
    pos = np.array(env.sim.data.site_xpos[site_id]).copy()
    mat = np.array(env.sim.data.site_xmat[site_id]).reshape(3, 3).copy()
    return pos, mat


def _get_handle_pose(env, source: str = "geom") -> Tuple[np.ndarray, np.ndarray]:
    handle_name = env.door.important_sites["handle"]
    if source == "geom":
        geom_id = env.sim.model.geom_name2id(handle_name)
        pos = np.array(env.sim.data.geom_xpos[geom_id]).copy()
        mat = np.array(env.sim.data.geom_xmat[geom_id]).reshape(3, 3).copy()
        return pos, mat

    site_id = env.sim.model.site_name2id(handle_name)
    pos = np.array(env.sim.data.site_xpos[site_id]).copy()
    mat = np.array(env.sim.data.site_xmat[site_id]).reshape(3, 3).copy()
    return pos, mat


def _get_handle_front_direction(env, handle_pos: np.ndarray, handle_mat: np.ndarray, arm: str) -> np.ndarray:
    eef_pos, _ = _get_eef_pose(env, arm=arm)
    direction = np.array(handle_mat[:, 1]).copy()
    direction[2] = 0.0
    if np.dot(direction, eef_pos - handle_pos) < 0:
        direction = -direction
    return _normalize(direction)


def _make_handle_front_orientation(approach_dir: np.ndarray, current_ori: np.ndarray) -> np.ndarray:
    desired_z = _normalize(-approach_dir)

    def build_with_axis(axis):
        desired_x = axis - np.dot(axis, desired_z) * desired_z
        if np.linalg.norm(desired_x) < 1e-6:
            desired_x = np.array([1.0, 0.0, 0.0])
            desired_x = desired_x - np.dot(desired_x, desired_z) * desired_z
        desired_x = _normalize(desired_x)
        desired_y = _normalize(np.cross(desired_z, desired_x))
        desired_x = _normalize(np.cross(desired_y, desired_z))
        return np.column_stack([desired_x, desired_y, desired_z])

    from robosuite.utils.control_utils import orientation_error

    candidates = [
        build_with_axis(np.array([0.0, 0.0, 1.0])),
        build_with_axis(np.array([0.0, 0.0, -1.0])),
    ]
    return min(candidates, key=lambda mat: np.linalg.norm(orientation_error(mat, current_ori)))


def _sync_controllers_to_current_pose(env) -> None:
    env.sim.forward()
    for robot in env.robots:
        for controller in robot.part_controllers.values():
            if hasattr(controller, "update"):
                controller.update(force=True)
        robot.composite_controller.update_state()
        robot.composite_controller.reset()


def _nominal_sampler_quat(sampler) -> np.ndarray:
    rotation = sampler.rotation
    if rotation is None:
        rot_angle = 0.0
    elif isinstance(rotation, (tuple, list, np.ndarray)):
        rot_angle = 0.5 * (min(rotation) + max(rotation))
    else:
        rot_angle = float(rotation)

    if sampler.rotation_axis == "x":
        return np.array([np.cos(rot_angle / 2), np.sin(rot_angle / 2), 0, 0])
    if sampler.rotation_axis == "y":
        return np.array([np.cos(rot_angle / 2), 0, np.sin(rot_angle / 2), 0])
    if sampler.rotation_axis == "z":
        return np.array([np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)])
    raise ValueError(f"Unsupported sampler rotation axis: {sampler.rotation_axis}")


def _set_door_to_nominal_sampler_pose(env) -> bool:
    if not hasattr(env, "door") or not hasattr(env, "placement_initializer"):
        return False

    sampler = env.placement_initializer
    required_attrs = ("x_range", "y_range", "reference_pos", "z_offset", "rotation_axis")
    if not all(hasattr(sampler, attr) for attr in required_attrs):
        return False

    from robosuite.utils.transform_utils import quat_multiply

    base_offset = np.asarray(sampler.reference_pos, dtype=float)
    door_pos = np.array(
        [
            base_offset[0] + 0.5 * (min(sampler.x_range) + max(sampler.x_range)),
            base_offset[1] + 0.5 * (min(sampler.y_range) + max(sampler.y_range)),
            base_offset[2] + sampler.z_offset - env.door.bottom_offset[-1],
        ]
    )
    door_quat = _nominal_sampler_quat(sampler)
    if hasattr(env.door, "init_quat"):
        door_quat = quat_multiply(door_quat, env.door.init_quat)

    door_body_id = env.sim.model.body_name2id(env.door.root_body)
    env.sim.model.body_pos[door_body_id] = door_pos
    env.sim.model.body_quat[door_body_id] = door_quat
    if hasattr(env, "_apply_latch_z_randomization"):
        env._apply_latch_z_randomization(randomize=False)
    env.sim.forward()
    return True


def _solve_arm_qpos_for_eef_pose(
    env,
    target_pos: np.ndarray,
    target_ori: np.ndarray,
    arm: str = "right",
    max_iters: int = 300,
    position_threshold: float = 0.012,
    orientation_threshold: float = 0.10,
) -> Tuple[int, float, float]:
    from robosuite.utils.control_utils import orientation_error

    robot = env.robots[0]
    site_name = env.sim.model.site_id2name(robot.eef_site_id[arm])
    qpos_indexes = np.asarray(robot._ref_arm_joint_pos_indexes)
    qvel_indexes = np.asarray(robot._ref_arm_joint_vel_indexes)
    joint_indexes = np.asarray(robot._ref_arm_joint_indexes)

    last_pos_err = np.inf
    last_ori_err = np.inf
    for step in range(max_iters):
        env.sim.forward()
        eef_pos, eef_ori = _get_eef_pose(env, arm=arm)
        pos_err = target_pos - eef_pos
        ori_err = orientation_error(target_ori, eef_ori)
        last_pos_err = float(np.linalg.norm(pos_err))
        last_ori_err = float(np.linalg.norm(ori_err))
        if last_pos_err < position_threshold and last_ori_err < orientation_threshold:
            break

        jac_pos = env.sim.data.get_site_jacp(site_name).reshape(3, -1)[:, qvel_indexes]
        jac_ori = env.sim.data.get_site_jacr(site_name).reshape(3, -1)[:, qvel_indexes]
        ori_weight = 0.45
        jac = np.vstack([jac_pos, ori_weight * jac_ori])
        err = np.concatenate([pos_err, ori_weight * ori_err])

        damping = 0.04
        dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * damping * np.eye(6), err)
        max_delta = np.max(np.abs(dq))
        if max_delta > 0.08:
            dq *= 0.08 / max_delta

        env.sim.data.qpos[qpos_indexes] += dq
        for joint_id, qpos_addr in zip(joint_indexes, qpos_indexes):
            if env.sim.model.jnt_limited[joint_id]:
                low, high = env.sim.model.jnt_range[joint_id]
                env.sim.data.qpos[qpos_addr] = np.clip(env.sim.data.qpos[qpos_addr], low, high)

    env.sim.data.qvel[qvel_indexes] = 0.0
    env.sim.forward()
    return step + 1, last_pos_err, last_ori_err


def _set_initial_eef_in_front_of_handle(
    env,
    distance: float,
    handle_source: str,
    orientation: str,
) -> Optional[Dict[str, Any]]:
    if not hasattr(env, "door"):
        return None

    arm = env.robots[0].arms[0]
    handle_pos, handle_mat = _get_handle_pose(env, source=handle_source)
    approach_dir = _get_handle_front_direction(env, handle_pos, handle_mat, arm=arm)
    target_pos = handle_pos + approach_dir * distance

    _, current_ori = _get_eef_pose(env, arm=arm)
    target_ori = current_ori if orientation == "current" else _make_handle_front_orientation(approach_dir, current_ori)

    _solve_arm_qpos_for_eef_pose(env, target_pos=target_pos, target_ori=target_ori, arm=arm)
    _sync_controllers_to_current_pose(env)

    robot = env.robots[0]
    return {
        "arm": arm,
        "arm_joint_names": list(robot.robot_model.arm_joints),
        "ready_qpos": np.array(env.sim.data.qpos[robot._ref_arm_joint_pos_indexes]).copy(),
        "target_pos": target_pos.copy(),
        "nominal_handle_pos": handle_pos.copy(),
        "distance": float(distance),
    }


def _apply_fixed_ready_state(env, ready_state: Optional[Dict[str, Any]]) -> None:
    if ready_state is None:
        return

    _set_mobile_base_qpos(env, ready_state.get("base_state"))

    arm_state = ready_state.get("arm_state")
    if arm_state is None:
        _sync_controllers_to_current_pose(env)
        return

    qpos_indexes = [env.sim.model.get_joint_qpos_addr(name) for name in arm_state["arm_joint_names"]]
    qvel_indexes = [env.sim.model.get_joint_qvel_addr(name) for name in arm_state["arm_joint_names"]]
    env.sim.data.qpos[qpos_indexes] = arm_state["ready_qpos"]
    env.sim.data.qvel[qvel_indexes] = 0.0
    env.sim.forward()
    _sync_controllers_to_current_pose(env)


def _resolve_sensor_name(env, suffix: str) -> str:
    names = [env.sim.model.sensor_id2name(i) for i in range(env.sim.model.nsensor)]
    if suffix in names:
        return suffix
    matches = [name for name in names if name is not None and name.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple sensors match suffix '{suffix}': {matches}")
    raise RuntimeError(f"Cannot find sensor ending with '{suffix}'. Available sensors: {names}")


def _read_mujoco_sensor(env, sensor_name: str) -> np.ndarray:
    sensor_id = env.sim.model.sensor_name2id(sensor_name)
    adr = int(env.sim.model.sensor_adr[sensor_id])
    dim = int(env.sim.model.sensor_dim[sensor_id])
    return np.asarray(env.sim.data.sensordata[adr : adr + dim]).copy()


class RobosuiteDoorHILSERLEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        config: Optional[RobosuiteDoorEnvConfig] = None,
        fake_env: bool = False,
        save_video: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.config = config or RobosuiteDoorEnvConfig()
        for key, value in kwargs.items():
            setattr(self.config, key, value)

        self.fake_env = fake_env
        self.save_video = save_video
        self.env = None
        self.rs_env = None
        self.prev_action = np.zeros(7, dtype=np.float32)
        self.step_count = 0
        self._last_rs_obs = None
        self._ready_state = None
        self._ft_force_sensor_name = None
        self._ft_torque_sensor_name = None
        self._arm = "right"
        self._arm_qpos_dim = int(self.config.arm_qpos_dim)
        self._gripper_qpos_dim = int(self.config.gripper_qpos_dim)

        self.action_space = gym.spaces.Box(
            low=np.asarray(self.config.action_low, dtype=np.float32),
            high=np.asarray(self.config.action_high, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = self._make_observation_space(self._arm_qpos_dim, self._gripper_qpos_dim)

        if not self.fake_env:
            self.env = self._make_robosuite_env()
            self.rs_env = self.env
            self._sync_dimensions_from_env()
            self.observation_space = self._make_observation_space(self._arm_qpos_dim, self._gripper_qpos_dim)
            self._ready_state = self._build_fixed_ready_state()
            self._reset_and_apply_ready_state()

    @property
    def robosuite_env(self):
        return self.rs_env

    def get_robosuite_env(self):
        return self.rs_env

    def _image_hw(self) -> Tuple[int, int]:
        height = getattr(self.config, "camera_heights", self.config.image_size[0])
        width = getattr(self.config, "camera_widths", self.config.image_size[1])
        return int(height), int(width)

    def _make_observation_space(self, arm_qpos_dim: int, gripper_qpos_dim: int) -> gym.spaces.Dict:
        image_h, image_w = self._image_hw()
        return gym.spaces.Dict(
            {
                "images": gym.spaces.Dict(
                    {
                        key: gym.spaces.Box(0, 255, shape=(image_h, image_w, 3), dtype=np.uint8)
                        for key in self.config.image_keys
                    }
                ),
                "state": gym.spaces.Dict(
                    {
                        "eef_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                        "eef_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                        "arm_qpos": gym.spaces.Box(-np.inf, np.inf, shape=(arm_qpos_dim,), dtype=np.float32),
                        "arm_qvel": gym.spaces.Box(-np.inf, np.inf, shape=(arm_qpos_dim,), dtype=np.float32),
                        "gripper_qpos": gym.spaces.Box(-np.inf, np.inf, shape=(gripper_qpos_dim,), dtype=np.float32),
                        "gripper_qvel": gym.spaces.Box(-np.inf, np.inf, shape=(gripper_qpos_dim,), dtype=np.float32),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                        "prev_action": gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32),
                    }
                ),
            }
        )

    def _zero_obs(self) -> Dict[str, Dict[str, np.ndarray]]:
        image_h, image_w = self._image_hw()
        return {
            "images": {
                key: np.zeros((image_h, image_w, 3), dtype=np.uint8)
                for key in self.config.image_keys
            },
            "state": {
                "eef_pose": np.zeros(7, dtype=np.float32),
                "eef_vel": np.zeros(6, dtype=np.float32),
                "arm_qpos": np.zeros(self._arm_qpos_dim, dtype=np.float32),
                "arm_qvel": np.zeros(self._arm_qpos_dim, dtype=np.float32),
                "gripper_qpos": np.zeros(self._gripper_qpos_dim, dtype=np.float32),
                "gripper_qvel": np.zeros(self._gripper_qpos_dim, dtype=np.float32),
                "tcp_force": np.zeros(3, dtype=np.float32),
                "tcp_torque": np.zeros(3, dtype=np.float32),
                "prev_action": self.prev_action.copy(),
            },
        }

    def _make_robosuite_env(self):
        os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

        import robosuite as suite
        from robosuite import load_composite_controller_config
        from robosuite.utils.robot_composition_utils import create_composite_robot

        if self.config.base is None:
            robots = [self.config.robot]
            controller_config = load_composite_controller_config(
                controller=self.config.controller,
                robot=self.config.robot,
            )
        else:
            composed_name = self.config.mobile_robot_name or f"{self.config.robot}{self.config.base}"
            create_composite_robot(
                name=composed_name,
                robot=self.config.robot,
                base=self.config.base,
                grippers=self.config.gripper,
            )
            robots = [composed_name]
            if self.config.controller is None:
                controller_config = copy.deepcopy(MOBILE_CONTROLLER_CONFIG)
            else:
                controller_config = load_composite_controller_config(
                    controller=self.config.controller,
                    robot=composed_name,
                )

        suite_config = {
            "env_name": self.config.env_name,
            "robots": robots,
            "controller_configs": controller_config,
        }
        if "TwoArm" in self.config.env_name:
            suite_config["env_configuration"] = self.config.env_configuration
        if "Door" in self.config.env_name:
            suite_config["handle_z_randomization"] = self.config.handle_z_randomization

        return suite.make(
            **suite_config,
            has_renderer=self.config.has_renderer,
            has_offscreen_renderer=self.config.has_offscreen_renderer,
            render_camera=self.config.render_camera,
            render_gpu_device_id=self.config.render_gpu_device_id,
            ignore_done=self.config.ignore_done,
            use_camera_obs=self.config.use_camera_obs,
            use_object_obs=self.config.use_object_obs,
            reward_shaping=self.config.reward_shaping,
            control_freq=self.config.control_freq,
            hard_reset=self.config.hard_reset,
            camera_names=list(self.config.camera_names),
            camera_heights=self.config.camera_heights,
            camera_widths=self.config.camera_widths,
            camera_depths=False,
        )

    def _sync_dimensions_from_env(self) -> None:
        robot = self.env.robots[0]
        self._arm = robot.arms[0]
        self._arm_qpos_dim = len(robot._ref_arm_joint_pos_indexes)
        self._gripper_qpos_dim = len(robot._ref_gripper_joint_pos_indexes[self._arm])

    def _build_fixed_ready_state(self) -> Dict[str, Any]:
        base_env = _get_unwrapped_env(self.env)
        old_deterministic_reset = base_env.deterministic_reset
        base_env.deterministic_reset = True
        try:
            self.env.reset()
            _set_door_to_nominal_sampler_pose(base_env)
            base_state = _place_mobile_base_in_front_of_table(
                base_env,
                table_distance=self.config.mobile_base_door_distance,
                yaw=self.config.mobile_base_yaw,
                lateral_offset=self.config.mobile_base_lateral_offset,
            )
            arm_state = None
            if not self.config.skip_handle_pregrasp_init:
                arm_state = _set_initial_eef_in_front_of_handle(
                    base_env,
                    distance=self.config.handle_pregrasp_distance,
                    handle_source=self.config.handle_pregrasp_source,
                    orientation=self.config.handle_pregrasp_orientation,
                )
            return {"base_state": base_state, "arm_state": arm_state}
        finally:
            base_env.deterministic_reset = old_deterministic_reset

    def _clear_robot_motion_and_ctrl(self) -> None:
        robot = self.env.robots[0]
        for attr in ("_ref_joint_vel_indexes", "_ref_arm_joint_vel_indexes"):
            if hasattr(robot, attr):
                self.env.sim.data.qvel[getattr(robot, attr)] = 0.0

        if self._arm in robot._ref_gripper_joint_vel_indexes:
            self.env.sim.data.qvel[robot._ref_gripper_joint_vel_indexes[self._arm]] = 0.0

        refs = _get_mobile_base_refs(self.env)
        if refs is not None:
            for addr in refs["qvel_addrs"].values():
                self.env.sim.data.qvel[addr] = 0.0

        self.env.sim.data.ctrl[:] = 0.0
        self.env.sim.forward()
        _sync_controllers_to_current_pose(self.env)

    def _reset_and_apply_ready_state(self):
        self.env.reset()
        _apply_fixed_ready_state(self.env, self._ready_state)
        self._clear_robot_motion_and_ctrl()
        self.prev_action = np.zeros(7, dtype=np.float32)
        self.step_count = 0
        self._last_rs_obs = None
        return None

    def _read_ft(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._ft_force_sensor_name is None:
            self._ft_force_sensor_name = _resolve_sensor_name(self.env, "force_ee")
            self._ft_torque_sensor_name = _resolve_sensor_name(self.env, "torque_ee")
        force = _read_mujoco_sensor(self.env, self._ft_force_sensor_name).astype(np.float32)
        torque = _read_mujoco_sensor(self.env, self._ft_torque_sensor_name).astype(np.float32)
        return force, torque

    # def _camera_image_from_obs(self, rs_obs: Dict[str, Any], camera_name: str) -> np.ndarray:
    #     image_key = f"{camera_name}_image"
    #     image_h, image_w = self._image_hw()
    #     if rs_obs is not None and image_key in rs_obs:
    #         image = rs_obs[image_key]
    #     else:
    #         image = self.env.sim.render(camera_name=camera_name, height=image_h, width=image_w)
    #         image = image[::-1]
    #     image = np.asarray(image)
    #     if image.dtype != np.uint8:
    #         image = np.clip(image, 0, 255).astype(np.uint8)
    #     if image.shape[-1] > 3:
    #         image = image[..., :3]
    #     return np.ascontiguousarray(image)

    def _camera_image_from_obs(self, rs_obs: Dict[str, Any], camera_name: str) -> np.ndarray:
        image_key = f"{camera_name}_image"
        image_h, image_w = self._image_hw()

        if rs_obs is not None and image_key in rs_obs:
            image = rs_obs[image_key]
        else:
            image = self.env.sim.render(camera_name=camera_name, height=image_h, width=image_w)

        image = np.asarray(image)

        # MuJoCo / OpenGL offscreen images are read bottom-up relative to
        # NumPy / OpenCV image coordinates. Flip once here so all downstream
        # wrist / side observations are human-readable and consistent.
        image = image[::-1]

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.shape[-1] > 3:
            image = image[..., :3]

        return np.ascontiguousarray(image)

    def _get_obs(self, rs_obs: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, np.ndarray]]:
        if self.fake_env:
            return self._zero_obs()

        from robosuite.utils.transform_utils import mat2quat

        robot = self.env.robots[0]
        arm = self._arm
        eef_pos, eef_mat = _get_eef_pose(self.env, arm=arm)
        force, torque = self._read_ft()

        images = {
            image_key: self._camera_image_from_obs(rs_obs, camera_name)
            for image_key, camera_name in zip(self.config.image_keys, self.config.camera_names)
        }
        state = {
            "eef_pose": np.concatenate([eef_pos, mat2quat(eef_mat)]).astype(np.float32),
            "eef_vel": np.asarray(robot.recent_ee_vel[arm].current[:6], dtype=np.float32).copy(),
            "arm_qpos": np.asarray(self.env.sim.data.qpos[robot._ref_arm_joint_pos_indexes], dtype=np.float32).copy(),
            "arm_qvel": np.asarray(self.env.sim.data.qvel[robot._ref_arm_joint_vel_indexes], dtype=np.float32).copy(),
            "gripper_qpos": np.asarray(
                self.env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes[arm]],
                dtype=np.float32,
            ).copy(),
            "gripper_qvel": np.asarray(
                self.env.sim.data.qvel[robot._ref_gripper_joint_vel_indexes[arm]],
                dtype=np.float32,
            ).copy(),
            "tcp_force": force.reshape(3).astype(np.float32),
            "tcp_torque": torque.reshape(3).astype(np.float32),
            "prev_action": self.prev_action.copy(),
        }
        return {"images": images, "state": state}

    def _robosuite_action(self, action: np.ndarray) -> np.ndarray:
        robot = self.env.robots[0]
        arm = self._arm
        action_dict = {
            arm: np.asarray(action[:6], dtype=np.float32),
            robot.get_gripper_name(arm): np.asarray([action[6]], dtype=np.float32),
            "base_mode": -1.0,
        }
        for part_name in ("base", getattr(robot, "torso", "torso")):
            if part_name in robot._action_split_indexes:
                start_idx, end_idx = robot._action_split_indexes[part_name]
                action_dict[part_name] = np.zeros(end_idx - start_idx, dtype=np.float32)
        return robot.create_action_vector(action_dict)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if self.fake_env:
            self.prev_action = np.zeros(7, dtype=np.float32)
            self.step_count = 0
            return self._get_obs(), {}

        rs_obs = self._reset_and_apply_ready_state()
        return self._get_obs(rs_obs), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(7)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if self.fake_env:
            self.prev_action = action.copy()
            self.step_count += 1
            truncated = self.step_count >= self.config.max_episode_steps
            return self._get_obs(), 0.0, False, truncated, {}

        rs_action = self._robosuite_action(action)
        rs_obs, _, _, _ = self.env.step(rs_action)
        self.prev_action = action.copy()
        self.step_count += 1
        truncated = self.step_count >= self.config.max_episode_steps
        return self._get_obs(rs_obs), 0.0, False, truncated, {}

    def render(self):
        if self.fake_env:
            return None
        return self.env.render()

    def close(self):
        if self.env is not None:
            self.env.close()


def unwrap_robosuite_door_env(env):
    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, RobosuiteDoorHILSERLEnv):
            return current
        seen.add(id(current))
        current = getattr(current, "env", None)
    raise TypeError("Could not find RobosuiteDoorHILSERLEnv in wrapper chain.")


def make_default_env_config(**overrides) -> RobosuiteDoorEnvConfig:
    config = RobosuiteDoorEnvConfig()
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_env_from_namespace(args: SimpleNamespace) -> RobosuiteDoorHILSERLEnv:
    config = make_default_env_config(**vars(args))
    return RobosuiteDoorHILSERLEnv(config=config)
