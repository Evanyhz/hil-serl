import os
import time
from dataclasses import dataclass

import gymnasium as gym

from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

from experiments.config import DefaultTrainingConfig
from experiments.robosuite_door.env import RobosuiteDoorEnvConfig, RobosuiteDoorHILSERLEnv


class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    def __init__(self, env, reward_classifier_func, target_hz=None):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs)
        return 0

    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = done or rew
        info["succeed"] = bool(rew)
        if self.target_hz is not None:
            time.sleep(max(0, 1 / self.target_hz - (time.time() - start_time)))
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["succeed"] = False
        return obs, info


@dataclass
class EnvConfig(RobosuiteDoorEnvConfig):
    max_episode_steps: int = 120


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist", "side"]
    classifier_keys = ["wrist", "side"]
    proprio_keys = [
        "eef_pose",
        "eef_vel",
        "arm_qpos",
        "arm_qvel",
        "gripper_qpos",
        "gripper_qvel",
        "tcp_force",
        "tcp_torque",
        "prev_action",
    ]
    checkpoint_period = 5000
    buffer_period = 1000
    random_steps = 0
    discount = 0.98
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"
    max_traj_length = EnvConfig.max_episode_steps

    def get_environment(
        self,
        fake_env=False,
        save_video=False,
        classifier=False,
        has_renderer=False,
        render_camera="mobilebase0_base_sideview",
        classifier_checkpoint_path=None,
        **env_kwargs,
    ):
        env_options = {
            "has_renderer": has_renderer,
            "render_camera": render_camera,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
            "use_object_obs": False,
            "ignore_done": True,
            "camera_names": ("robot0_eye_in_hand", "mobilebase0_base_sideview"),
            "camera_heights": 128,
            "camera_widths": 128,
        }
        env_options.update(env_kwargs)
        env = RobosuiteDoorHILSERLEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
            **env_options,
        )
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        if classifier:
            import jax
            import jax.numpy as jnp

            from serl_launcher.networks.reward_classifier import load_classifier_func

            classifier_func = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath(classifier_checkpoint_path or "classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                return int(sigmoid(classifier_func(obs)) > 0.7)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env
