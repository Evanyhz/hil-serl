# HIL-SERL 开门任务复现交接说明

## 1. 当前目标

本项目目标是基于 **HIL-SERL 思路**，先在 robosuite / MuJoCo 中跑通移动机械臂开门任务，再逐步迁移到真实机械臂。

当前优先级：

1. 在 robosuite 中搭建可遥控的开门环境。
2. 用键盘完成底盘移动、机械臂靠近、抓取门把手、旋拧把手、推门。
3. 为后续 HIL-SERL 的数据采集、人工干预、奖励设计和策略训练提供基础环境。

---

## 2. 已准备好的仓库和环境

### HIL-SERL 仓库

已新建仓库：

```bash
hil-serl
```

### Conda 环境

已根据 `hil-serl` 仓库 README 配置好环境：

```bash
conda activate hilserl
```

后续复现 HIL-SERL 相关代码时，优先在该环境中操作。

---

## 3. robosuite 仿真侧已有基础

当前主要工作在 robosuite 仓库中完成：

```bash
~/robosuite
```
已有conda 环境
```bash
conda activate robosuite
```


主要 demo 文件：

```text
robosuite/demos/demo_device_control_mobile_base.py
```

当前任务配置：

```text
Environment: Door
Robot: Fairino5V6
Mobile base: OmronMobileBase
Gripper: Robotiq85Gripper
```

常用启动命令：

```bash
python robosuite/demos/demo_device_control_mobile_base.py \
  --environment Door \
  --robots Fairino5V6 \
  --base OmronMobileBase \
  --gripper Robotiq85Gripper \
  --render-camera mobilebase0_base_sideview \
  --print-cameras
```

腕部相机视角：

```bash
python robosuite/demos/demo_device_control_mobile_base.py \
  --environment Door \
  --robots Fairino5V6 \
  --base OmronMobileBase \
  --gripper Robotiq85Gripper \
  --render-camera robot0_eye_in_hand \
  --print-cameras
```

---

## 4. 已完成且对复现有帮助的内容

### 4.1 移动底盘遥控

已在 `demo_device_control_mobile_base.py` 中加入移动底盘键盘遥控。

控制键：

```text
I / K：前进 / 后退
J / L：左右平移
U / O：原地旋转
T / G：升降柱上 / 下
```

该功能用于将机器人移动到门把手前方。

---

### 4.2 reset 后自动摆放底盘

已实现：

```python
place_mobile_base_in_front_of_table(...)
```

作用：

1. 根据 `env.table_offset` 获取桌子 / 门环境中心。
2. 将 Omron 底盘摆到门前方。
3. 设置底盘 yaw，使其朝向门。
4. 清零底盘速度和控制输入。

这有利于稳定复现实验初始状态。

---

### 4.3 reset 后，机械臂每次预抓取的位置都固定

demo_device_control_mobile_base.py (line 145)
新增 set_door_to_nominal_sampler_pose()：第一次计算 ready_qpos 前，临时把 Door 放到 sampler 的 nominal 中心位姿。
新增 get_fixed_ready_state() / apply_fixed_ready_state()：第一次用 nominal 门把手前方约 10 cm 计算一次 FR5 ready_qpos，之后每次 reset 只恢复这个固定姿态，不再根据随机后的门把手重新 IK。
reset_mobile_demo() 已改成：缓存 fixed ready state -> 正常随机 reset 门 -> 恢复固定底盘和固定机械臂姿态。
新增参数 --handle-z-randomization，默认 0.015，表示把手高度 ±1.5 cm 随机。
新参数 --mobile-base-door-distance 替代语义上的 table distance；旧的 --mobile-base-table-distance 仍保留为别名。

door.py (line 423)
新增 handle_z_randomization。
保留 Door 原有 x / y / yaw 随机化。
新增 latch body 的 Z 随机，而不是只移动 handle site，保证观测点和真实接触几何一致。
注释已说明：门随机化用于制造把手相对固定机器人初始位姿的偏差。

large_door.py (line 64)
复用同一个 latch Z 随机逻辑；LargeDoor 仍不新增 x/y/yaw 随机。

---

### 4.4 新增底盘第三视角相机

已在 Omron mobile base XML 中添加固定第三视角相机。

文件：

```text
robosuite/models/assets/bases/omron_mobile_base.xml
```

XML 原始相机名：

```text
base_sideview
```

robosuite 运行时名称通常为：

```text
mobilebase0_base_sideview
```

相机结构：

1. 细白色 L 形支架。
2. 从底盘侧面横向伸出。
3. 横杆末端竖直向上。
4. 顶部安装小型相机模型。
5. 相机固定在底盘上，不自动对准门把手。

相机真实位置应放在镜头处，而不是相机机身中心：

```xml
<camera
    mode="fixed"
    name="base_sideview"
    pos="0 0 -0.026"
    fovy="68"
/>
```

这样可以避免相机模型自身进入画面。

---

### 4.5 腕部相机已存在

FR5 模型中已自带腕部相机。

文件：

```text
robosuite/models/assets/robots/fairino5_v6/robot.xml
```

XML 原始名称：

```text
eye_in_hand
```

robosuite 运行时名称：

```text
robot0_eye_in_hand
```

建议 HIL-SERL 复现时使用：

```text
wrist camera: robot0_eye_in_hand
third-person camera: mobilebase0_base_sideview
```

---

### 4.6 末端六维力传感器
robosuite/demos/demo_ft_sensor.py 中，实现了六维力的可视化；
数据已经过验证，可以拿来直接用。


### 4.7 当前控制器经验

当前机械臂使用 robosuite 的 `OSC_POSE` 控制器。

关键配置：

```python
"type": "OSC_POSE"
"input_type": "delta"
"input_ref_frame": "base"
"impedance_mode": "fixed"
```

---

### 4.8 键盘遥控和旋拧问题

当前先用键盘跑通项目，但要注意：

普通键盘控制的 roll / pitch / yaw 单轴末端姿态变化，不等价于绕门把手转轴旋拧。

已观察到的问题：
```text
抓住门把手后，用末端自旋可以把把手压下去；
但松开键盘后，机械臂和把手会回退一段角度。
```
原因：
```text
末端自旋轴和门把手实际旋转轴不一致。
键盘单轴控制是在强行扭夹爪和把手之间的接触约束。
松开键盘后，接触误差释放，因此产生回退。
```
---



## 5. 下一步任务
1. 在 `~/hil-serl/examples/experiments/robosuite_door/` 下新建任务目录，至少包含：

```text
env.py
config.py
test_env.py
train.py
```

2. 在 `env.py` 中实现 `RobosuiteDoorHILSERLEnv`，封装 robosuite Door 环境为 Gymnasium 接口：

```python
obs, info = env.reset()
obs, reward, done, truncated, info = env.step(action)
```

训练封装写在 `hil-serl` 工作空间，不继续写进 robosuite demo。

3. `reset()` 只恢复已经缓存好的固定初始状态，不再重新计算任何真值：

```text
调用 robosuite env.reset()
Door 保持已有 x / y / yaw 随机化
latch / handle 保持 z 随机化
恢复已缓存的移动底盘 qpos
恢复已缓存的 FR5 ready_qpos
不读取随机后的 handle pose
不重新 IK
不调用 set_door_to_nominal_sampler_pose()
清零底盘、机械臂、夹爪速度
清零相关 actuator ctrl
sync_controllers_to_current_pose(env)
prev_action = 0
step_count = 0
返回 HIL-SERL obs
```

4. 训练阶段完全不用仿真真值。policy obs、reward、done、info 中都不包含：

```text
door hinge qpos
handle qpos
handle pose
door pose
success flag
true contact point
true contact force
handle-to-eef relative pose
door-to-eef relative pose
```

5. observation 格式固定为：

```python
obs = {
    "images": {
        "wrist": wrist_image,
        "side": side_image,
    },
    "state": {
        "eef_pose": ...,
        "eef_vel": ...,
        "arm_qpos": ...,
        "arm_qvel": ...,
        "gripper_qpos": ...,
        "gripper_qvel": ...,
        "tcp_force": ...,
        "tcp_torque": ...,
        "prev_action": ...,
    },
}
```

图像要求：

```text
wrist = robot0_eye_in_hand
side  = mobilebase0_base_sideview
shape = (128, 128, 3)
dtype = np.uint8
range = 0 ~ 255
format = RGB
```

六维力要求：

```text
tcp_force  = force_ee
tcp_torque = torque_ee
dtype = np.float32
shape = (3,)
```

仿真阶段直接使用 robosuite / MuJoCo 自带六维力传感器，不额外做复杂滤波或真值估计。

6. action 空间固定为 7 维：

```text
dx, dy, dz, droll, dpitch, dyaw, gripper
```

wrapper 内部映射到 robosuite composite action：

```text
right arm OSC_POSE delta = action[:6]
gripper command = action[6]
base action = 0
torso action = 0
```

移动底盘和升降柱只用于 reset 初始化和人工调试，不进入 policy action。

7. 在 `config.py` 中参考官方 `ram_insertion` 和 `usb_pickup_insertion`，设置：

```python
image_keys = ["wrist", "side"]

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

classifier_keys = ["wrist", "side"]
```

wrapper 链路：

```python
env = RobosuiteDoorHILSERLEnv(...)
env = SERLObsWrapper(env, proprio_keys=proprio_keys)
env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
env = MultiCameraBinaryRewardClassifierWrapper(env, reward_classifier_func)
```

8. `RobosuiteDoorHILSERLEnv.step()` 中 reward 和 done 只做占位：

```python
reward = 0.0
done = False
truncated = step_count >= max_episode_steps
info = {}
```

正式训练 reward 和 done 由 `MultiCameraBinaryRewardClassifierWrapper` 根据 `classifier_keys = ["wrist", "side"]` 输出，不使用仿真真值。

9. 在 `test_env.py` 中验证：

```text
reset 后底盘 qpos 固定
reset 后 FR5 ready_qpos 固定
不重新 IK 对准随机后的门把手
Door x / y / yaw 随 reset 变化
handle z 随 reset 变化且不累加漂移
obs 图像 shape / dtype / min / max 正常
tcp_force / tcp_torque shape 正确
policy obs 不含任何仿真真值字段
action_space.shape == (7,)
step() 返回格式正确
零 action 下底盘和机械臂不漂移
```

10. `train.py` 负责接入 HIL-SERL 官方训练流程，使用奖励分类器产生 reward；不要在 robosuite 环境里写基于 hinge qpos / handle qpos 的 reward 或 success。
