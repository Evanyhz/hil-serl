hil-serl/
├── README.md
├── docs/
│   ├── franka_walkthrough.md
│   └── images/
│       ├── task_banner.gif
│       └── robot_infra_interfaces.png
│
├── examples/
│   ├── train_rlpd.py
│   ├── record_demos.py
│   ├── record_success_fail.py
│   ├── train_reward_classifier.py
│   │
│   └── experiments/
│       ├── config.py
│       ├── mappings.py
│       │
│       ├── ram_insertion/
│       │   ├── config.py
│       │   ├── wrapper.py
│       │   ├── run_actor.sh
│       │   ├── run_learner.sh
│       │   ├── classifier_data/        # 运行后生成
│       │   ├── classifier_ckpt/        # 运行后生成
│       │   ├── demo_data/              # 运行后生成
│       │   └── videos/                 # 可选，运行后生成
│       │
│       ├── usb_pickup_insertion/
│       │   ├── config.py
│       │   ├── wrapper.py
│       │   ├── run_actor.sh
│       │   ├── run_learner.sh
│       │   ├── classifier_data/
│       │   ├── classifier_ckpt/
│       │   └── demo_data/
│       │
│       ├── object_handover/
│       │   ├── config.py
│       │   ├── wrapper.py
│       │   ├── run_actor.sh
│       │   └── run_learner.sh
│       │
│       └── egg_flip/
│           ├── config.py
│           ├── wrapper.py
│           ├── run_actor.sh
│           └── run_learner.sh
│
├── serl_launcher/
│   ├── setup.py
│   ├── requirements.txt
│   │
│   └── serl_launcher/
│       ├── agents/
│       │   └── continuous/
│       │       ├── bc.py
│       │       ├── sac.py
│       │       ├── sac_hybrid_single.py
│       │       └── sac_hybrid_dual.py
│       │
│       ├── common/
│       │   ├── common.py
│       │   ├── encoding.py
│       │   ├── optimizers.py
│       │   ├── typing.py
│       │   └── wandb.py
│       │
│       ├── data/
│       │   ├── replay_buffer.py
│       │   ├── memory_efficient_replay_buffer.py
│       │   └── data_store.py
│       │
│       ├── networks/
│       │   ├── actor_critic_nets.py
│       │   ├── lagrange.py
│       │   ├── mlp.py
│       │   └── reward_classifier.py
│       │
│       ├── utils/
│       │   ├── launcher.py
│       │   ├── timer_utils.py
│       │   └── train_utils.py
│       │
│       ├── vision/
│       │   ├── data_augmentations.py
│       │   └── resnet_v1.py
│       │
│       └── wrappers/
│           ├── chunking.py
│           └── serl_obs_wrappers.py
│
└── serl_robot_infra/
    ├── README.md
    ├── setup.py
    │
    ├── franka_env/
    │   ├── camera/
    │   │   ├── rs_capture.py
    │   │   └── video_capture.py
    │   │
    │   ├── envs/
    │   │   ├── franka_env.py
    │   │   ├── wrappers.py
    │   │   └── relative_env.py
    │   │
    │   ├── spacemouse/
    │   │   └── spacemouse_expert.py
    │   │
    │   └── utils/
    │       └── rotations.py
    │
    └── robot_servers/
        ├── franka_server.py
        ├── franka_gripper_server.py
        ├── robotiq_gripper_server.py
        ├── launch_left_server.sh
        └── launch_right_server.sh


1. 项目总体定位

hil-serl 不是单纯的仿真强化学习项目，而是一个偏真实机器人训练的 HIL-SERL 框架：它提供环境 wrapper、RL agent、replay buffer、视觉模块、reward classifier、Franka 机器人控制服务和若干真实任务例子。README 里明确说它用于“结合 demonstrations 和 human corrections 训练机器人操作策略”，目标是接近高成功率的真实机器人 manipulation。项目主结构也被 README 分成 examples、serl_launcher、serl_robot_infra 三块。

从复现角度看，三块分别对应：

examples             复现实验入口：采集数据、训练 reward classifier、actor/learner 训练
serl_launcher        算法库：SAC / BC / replay buffer / 网络 / wrapper / wandb / agentlace
serl_robot_infra     真机接口：Franka Gym 环境、相机、spacemouse、ROS + Flask 控制服务

项目训练架构是异步 actor-learner。README 说明 actor 和 learner 都与 robot gym environment 交互，actor 通过网络把数据发给 learner，learner 周期性同步策略到 actor；通信依赖 agentlace。

2. 安装结构
2.1 Python / JAX / serl_launcher

README 推荐先创建 hilserl conda 环境，Python 版本为 3.10；JAX 根据 CPU/GPU/TPU 分别安装，其中 GPU 示例使用 CUDA 12 对应的 jax[cuda12_pip]==0.4.35。随后进入 serl_launcher，执行 pip install -e . 和 pip install -r requirements.txt。

serl_launcher/setup.py 定义包名是 serl_launcher，核心依赖包括 zmq、opencv-python、lz4，以及固定 commit 的 agentlace。

serl_launcher/requirements.txt 里能看到主要训练栈：gym/gymnasium、numpy、flax、optax、distrax、tensorflow、tensorflow_probability、wandb、imageio、moviepy、pynput、matplotlib 等。

2.2 serl_robot_infra

serl_robot_infra 是真机侧依赖。它需要先安装 libfranka、franka_ros，再安装 serl_franka_controllers，然后在 serl_robot_infra 下 pip install -e .。

它自己的 setup.py 依赖包括 pyrealsense2、pymodbus==2.5.3、pyspacemouse、hidapi、rospkg、requests、flask 等，说明它不是纯 Python 离线训练库，而是要接 RealSense、SpaceMouse、ROS、HTTP 控制服务和真机。

3. examples/：复现实验最重要的入口

examples 下面的 4 个顶层脚本是复现主线：

train_rlpd.py                actor / learner 训练入口
record_success_fail.py       采集 reward classifier 的成功/失败图像数据
train_reward_classifier.py   训练视觉 reward classifier
record_demos.py              用 SpaceMouse 采集少量成功 demonstrations
3.1 experiments/mappings.py

所有任务通过 CONFIG_MAPPING 注册。目前映射了四个实验：

ram_insertion
usb_pickup_insertion
object_handover
egg_flip

入口脚本会根据 --exp_name 找对应的 TrainConfig。

这意味着你复现或改任务时，通常要新增：

examples/experiments/your_task/
├── config.py
├── wrapper.py
├── run_actor.sh
└── run_learner.sh

然后把它注册到 experiments/mappings.py。

3.2 experiments/config.py

这里定义默认训练超参数，例如：

batch_size = 256
cta_ratio = 2
discount = 0.97
max_steps = 1,000,000
replay_buffer_capacity = 200,000
training_starts = 100
steps_per_update = 50
encoder_type = "resnet-pretrained"
setup_mode = "single-arm-fixed-gripper"

还规定每个任务必须实现 get_environment()。

setup_mode 对复现很关键，它决定用哪种 agent：

single-arm-fixed-gripper      固定夹爪，不学习 gripper
single-arm-learned-gripper    单臂，学习 gripper
dual-arm-fixed-gripper        双臂，固定夹爪
dual-arm-learned-gripper      双臂，学习 gripper
4. train_rlpd.py：actor / learner 主训练逻辑

这是整个项目最重要的训练入口。

4.1 actor 逻辑

actor() 在 --actor 模式下运行。它负责：

env.reset()
sample action
env.step(action)
记录 transition
如果人类 intervention 发生，用 intervene_action 替换策略动作
把普通数据送入 actor_env replay store
把 intervention 数据额外送入 actor_env_intvn store
episode 结束后把统计信息发给 learner
周期性保存 buffer

代码中 actor 会检测 info["intervene_action"]，如果存在，就把当前动作替换成人类干预动作，同时记录 intervention 次数和步数。

actor 还会创建 TrainerClient，并注册接收 learner 下发网络参数的 callback。

4.2 learner 逻辑

learner() 在 --learner 模式下运行。它创建 TrainerServer，注册两个数据源：

actor_env         普通在线 replay buffer
actor_env_intvn   人类 intervention buffer

然后等待 replay buffer 达到 training_starts，再开始训练。

训练时采用 50/50 采样：一半来自在线 replay buffer，一半来自 demo/intervention buffer。这个是 HIL-SERL 复现时非常关键的点。

learner 中还可以看到 critic 多次更新、actor/temperature 周期性更新、周期性向 actor 发布网络参数、周期性保存 checkpoint。

4.3 agent 选择

main() 根据 setup_mode 选择 agent：

single-arm-fixed-gripper / dual-arm-fixed-gripper
    -> make_sac_pixel_agent

single-arm-learned-gripper
    -> make_sac_pixel_agent_hybrid_single_arm

dual-arm-learned-gripper
    -> make_sac_pixel_agent_hybrid_dual_arm

也就是说，如果你的开门任务一开始希望“把手已经被抓住，夹爪固定闭合”，应该更接近 single-arm-fixed-gripper；如果希望策略还学习开合夹爪，就需要走 hybrid agent。

5. record_success_fail.py：reward classifier 数据采集

这个脚本用于采集成功/失败图像样本。它会创建环境，但 classifier=False，也就是采集 classifier 的训练数据时先不使用 classifier。默认成功样本数是 200。运行时按空格键标记当前 transition 为成功，否则默认进入失败集。最后分别保存 success_images 和 failure_images 的 pkl 文件。

复现时的含义是：你需要先手工操作机器人到成功/失败状态，采集视觉分类器数据。对于你未来的开门任务，这一步大概率对应：

成功样本：门把手已转动到位 / 门已打开到阈值
失败样本：抓偏、没转动、半开、误触、碰门板、手在空中等
6. train_reward_classifier.py：训练视觉奖励模型

这个脚本读取当前任务目录下的：

classifier_data/*success*.pkl
classifier_data/*failure*.pkl

然后构建正负 replay buffer，正样本标签为 1，负样本标签为 0。训练时每个 batch 一半正样本、一半负样本，并对 classifier_keys 对应图像做随机裁剪增强。训练好的 classifier 保存到 classifier_ckpt/。

这说明项目默认不是用稠密 reward，而是依赖图像二分类器作为 sparse reward。对于真实开门，复现难点之一就是 reward classifier 是否稳，尤其是“半成功”和“视觉遮挡”的负样本要采够。

7. record_demos.py：采集 demonstrations

record_demos.py 默认采集 20 条成功 demo。每步动作初始为零，如果 SpaceMouse 干预，则从 info["intervene_action"] 读取人工动作。只有当 episode 成功时，整条 trajectory 才会被加入 demonstration 数据。最后保存到 demo_data/。

复现时的关键点是：demo 不是单步成功帧，而是完整 transition 序列。learner 后续会把这些 demo 加入 demo buffer，并和在线数据 50/50 混合训练。

8. 具体任务配置：以 RAM insertion 为例

ram_insertion/config.py 体现了一个完整任务需要配置什么：

SERVER_URL
REALSENSE_CAMERAS
IMAGE_CROP
TARGET_POSE
GRASP_POSE
RESET_POSE
ABS_POSE_LIMIT_LOW / HIGH
RANDOM_RESET
ACTION_SCALE
COMPLIANCE_PARAM
PRECISION_PARAM
image_keys
classifier_keys
proprio_keys
encoder_type
setup_mode
get_environment()

RAM 任务中使用两个 wrist camera；TARGET_POSE / GRASP_POSE / RESET_POSE 是人工采集的真实机器人位姿；ABS_POSE_LIMIT_LOW / HIGH 定义策略安全探索边界；COMPLIANCE_PARAM / PRECISION_PARAM 定义阻抗控制参数。

TrainConfig.get_environment() 的 wrapper 顺序也非常重要：

RAMEnv
-> GripperCloseEnv
-> SpacemouseIntervention
-> RelativeFrame
-> Quat2EulerWrapper
-> SERLObsWrapper
-> ChunkingWrapper
-> MultiCameraBinaryRewardClassifierWrapper

其中 SpacemouseIntervention 只在 fake_env=False 时启用，也就是真机 actor/数据采集时启用，learner 的 fake env 不启用。classifier 模式下，会加载 classifier_ckpt/ 并用图像分类器输出 reward。

ram_insertion/wrapper.py 还自定义了 reset 和 regrasp。它会在 reset 前切换到 precision 参数、上提避免碰撞、随机化 reset 位姿，然后切回 compliance 参数。按 F1 时会触发重新抓取 RAM。

9. USB pickup insertion 任务的差异

USB 任务更接近“夹爪也学习”的情况。它设置：

setup_mode = "single-arm-learned-gripper"
image_keys = ["side_policy", "wrist_1", "wrist_2"]
classifier_keys = ["side_classifier"]

这里有一个细节：同一个 RealSense 相机被拆成 side_policy 和 side_classifier 两个不同 crop，一个给策略，一个给 reward classifier。

对你的开门任务有启发：策略图像和 reward classifier 图像不一定要相同。策略可以看宽视角，reward classifier 可以看门缝、门把手角度、门板开度等更局部的位置。

10. serl_launcher/：算法库
10.1 utils/launcher.py

这个文件负责创建 agent、数据增强函数、TrainerConfig、wandb logger。它支持：

make_bc_agent
make_sac_pixel_agent
make_sac_pixel_agent_hybrid_single_arm
make_sac_pixel_agent_hybrid_dual_arm
make_trainer_config
make_wandb_logger

其中 SAC pixel agent 默认使用图像 encoder、proprioception、SAC actor-critic 网络和随机 crop 数据增强。

make_trainer_config() 默认端口是：

port_number = 5588
broadcast_port = 5589
request_types = ["send-stats"]

这就是 actor / learner 通信的默认端口。

10.2 agents/continuous/sac.py

SACAgent 是核心 RL agent。它包含：

forward_critic
forward_target_critic
forward_policy
critic_loss_fn
policy_loss_fn
temperature_loss_fn
update
sample_actions
create_pixels

create_pixels() 会根据 encoder_type 创建 ResNet encoder；resnet-pretrained 会构建 frozen ResNet10，并在最后加载预训练参数。

update() 中会先解包 memory efficient buffer，然后做图像数据增强，再分别更新 critic / actor / temperature，并软更新 target critic。

10.3 data/data_store.py

这里把 replay buffer 包装成 agentlace 可以使用的 DataStore，并加了线程锁，避免 actor/learner 异步读写时出问题。MemoryEfficientReplayBufferDataStore 是训练主脚本里实际使用的 buffer。

10.4 wrappers/

SERLObsWrapper 把原始 observation 整理成：

{
  "state": flattened proprioception,
  image_key_1: image,
  image_key_2: image,
  ...
}

它会从原始 obs["state"] 中选取 proprio_keys，然后 flatten；图像从 obs["images"] 里拿出来平铺到顶层。

ChunkingWrapper 支持 observation history 和 receding horizon action。当前配置里常用 obs_horizon=1、act_exec_horizon=None，也就是不堆叠历史，只包一层时间维。

11. serl_robot_infra/：真机与 Franka 环境
11.1 整体机制

serl_robot_infra/README.md 明确说：机器人侧有一个 Flask server，通过 ROS 给机器人发命令；gym env 通过 HTTP post 和 Flask server 通信。

所以它的控制链路是：

train_rlpd.py / record_demos.py
        |
        v
FrankaEnv.step()
        |
        v
HTTP POST
        |
        v
robot_servers/franka_server.py
        |
        v
ROS / serl_franka_controllers
        |
        v
Franka robot
11.2 franka_server.py

franka_server.py 启动 Flask server，并启动 ROS impedance controller。它的类 FrankaServer 会发布 /cartesian_impedance_controller/equilibrium_pose，订阅 Franka state 和 Jacobian，并通过 ROS launch 启停 impedance controller。

启动时支持这些参数：

--robot_ip
--gripper_ip
--gripper_type Robotiq|Franka|None
--reset_joint_target
--flask_url
--ros_port

这些参数在 serl_robot_infra/README.md 里也作为启动命令的一部分出现。

README 列出了 server 支持的 HTTP 请求，包括：

startimp / stopimp
pose
getpos / getvel / getforce / gettorque
getq / getdq / getjacobian / getstate
jointreset
activate_gripper / reset_gripper
get_gripper / close_gripper / open_gripper / move_gripper
clearerr
update_param

这些接口就是 FrankaEnv 控制机器人和读取状态的基础。

11.3 franka_env.py

FrankaEnv 是 gym 环境核心。它定义了：

action_space: 7 维 [-1, 1]
    action[:3]      xyz delta
    action[3:6]     rotvec delta
    action[6]       gripper action

observation_space:
    state:
        tcp_pose
        tcp_vel
        gripper_pose
        tcp_force
        tcp_torque
    images:
        每个 RealSense camera 一张 128x128 RGB 图像

代码里 action space 是 7 维，observation 包含 tcp pose、velocity、force、torque、gripper 和多相机图像。

step() 会裁剪 action，按 ACTION_SCALE 转成末端位姿增量，发送 gripper 命令和末端 pose 命令，然后读取新状态、计算 reward、判断 done。

图像由 RealSense 读取，先按任务配置 crop，再 resize 到 observation space 的 128×128。

reset 时会调用 update_param 切换阻抗参数、执行安全恢复、回到 reset pose，然后重新取 observation。

11.4 franka_env/envs/wrappers.py

这个文件很关键，包含人类干预、reward wrapper、姿态转换、夹爪动作处理等。

主要 wrapper：

HumanClassifierWrapper
MultiCameraBinaryRewardClassifierWrapper
MultiStageBinaryRewardClassifierWrapper
Quat2EulerWrapper
Quat2R2Wrapper
DualQuat2EulerWrapper
GripperCloseEnv
SpacemouseIntervention
DualSpacemouseIntervention
GripperPenaltyWrapper

其中 MultiCameraBinaryRewardClassifierWrapper 会用图像 reward classifier 替换环境 reward，并在 reward 为真时结束 episode。

SpacemouseIntervention 是 HIL 的核心。它读取 SpaceMouse 动作，如果人工动作非零，就替代策略动作，并把替代动作写入 info["intervene_action"]，后续 actor 会把它作为人类干预数据存下来。

12. 官方复现流程

以 RAM insertion 为例，官方 walkthrough 给出的流程是：

1. 安装 Python、Franka 控制器、serl_robot_infra
2. 启动 Franka server
3. 修改实验 config.py
4. 配置相机 serial number 和 crop
5. 采集 TARGET_POSE / GRASP_POSE / RESET_POSE
6. 采集 reward classifier 成功/失败数据
7. 训练 reward classifier
8. 采集 20 条 demonstrations
9. 同时启动 actor 和 learner 训练
10. 训练过程中用 SpaceMouse 人工干预
11. 指定 checkpoint 做评估

walkthrough 中明确建议 RAM insertion 先读，因为它包含完整训练和评估流程；其中 reward classifier 数据采集用 record_success_fail.py，demo 采集用 record_demos.py，训练用任务目录下的 run_actor.sh 和 run_learner.sh。

RAM 的 run_actor.sh 实际就是调用：

python ../../train_rlpd.py \
    --exp_name=ram_insertion \
    --checkpoint_path=first_run \
    --actor

并设置 JAX 显存预分配参数。

RAM 的 run_learner.sh 调用：

python ../../train_rlpd.py \
    --exp_name=ram_insertion \
    --checkpoint_path=first_run \
    --demo_path=... \
    --learner

所以 learner 必须传入 demo 数据路径。

13. 对你复现/改开门任务的直接启发

你后续要在 HIL-SERL 上做开门，不应该先动 serl_launcher 的 SAC 算法主体，而应该先仿照 examples/experiments/ram_insertion 或 usb_pickup_insertion 新建任务目录。

推荐任务结构：

examples/experiments/door_opening/
├── config.py
├── wrapper.py
├── run_actor.sh
└── run_learner.sh

然后在：

examples/experiments/mappings.py

里加入：

"door_opening": DoorOpeningTrainConfig

对开门任务，最核心要改的是：

EnvConfig.SERVER_URL
EnvConfig.REALSENSE_CAMERAS
EnvConfig.IMAGE_CROP
EnvConfig.TARGET_POSE
EnvConfig.RESET_POSE
EnvConfig.ABS_POSE_LIMIT_LOW / HIGH
EnvConfig.ACTION_SCALE
EnvConfig.COMPLIANCE_PARAM / PRECISION_PARAM
TrainConfig.image_keys
TrainConfig.classifier_keys
TrainConfig.proprio_keys
TrainConfig.setup_mode
DoorOpeningEnv.reset()
DoorOpeningEnv.go_to_reset()
reward_func()

你的设想“YOLO 只负责第一步接近和粗对准，RL 只从受限初始分布开始学开门”非常适合接到这个项目的结构里：YOLO/SAM/点云模块可以放在 reset 前或 episode 开始前，把机械臂带到门把手附近；HIL-SERL 的任务环境只负责从“已接近、已粗对准、甚至已抓住”的状态继续学习转动和推门。这样可以直接利用 HIL-SERL 已有的 ABS_POSE_LIMIT、RANDOM_RESET、SpacemouseIntervention、reward classifier 和 actor/learner 训练框架。

对于第一版复现，我建议优先走：

single-arm-fixed-gripper

也就是假设已经抓住把手，策略只学习末端 6D 增量动作，不学习夹爪开合。这和 RAM insertion 的 GripperCloseEnv + single-arm-fixed-gripper 更接近，训练难度明显低于 USB 那种 learned gripper。RAM 配置就是固定夹爪模式，且用 GripperCloseEnv 把 7 维动作压成 6 维末端动作。

14. 一句话总结

这个项目的复现主线不是“先看 SAC 算法”，而是：

启动 Franka server
→ 配好任务 config
→ 采集 reward classifier 数据
→ 训练 classifier
→ 采集少量 demonstrations
→ actor / learner 异步训练
→ SpaceMouse 人工干预加速收敛
→ checkpoint 评估

你做开门任务时，最应该优先复制和改造的是：

examples/experiments/ram_insertion/
serl_robot_infra/franka_env/envs/franka_env.py
serl_robot_infra/franka_env/envs/wrappers.py
examples/train_rlpd.py

其中 train_rlpd.py 和 serl_launcher 尽量先不要改，先把新任务包装成符合它预期的 gym environment。