from __future__ import annotations

import math

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp


"""
最小 Isaac Lab Go2 速度跟踪环境。

这个文件的目标不是复现 PASIST 全部逻辑，而是先提供一个“能启动、能训练、
能理解 Isaac Lab 环境结构”的最小环境配置。

它复用 Unitree RL Lab 中已经写好的：
- Go2 机器人资产配置: UNITREE_GO2_CFG
- locomotion MDP 函数: unitree_rl_lab.tasks.locomotion.mdp

它保留的最小环境闭环是：
1. 场景中放一个 Go2 和一块平地
2. command manager 采样目标 base velocity
3. policy 输出关节位置控制 action
4. observation manager 生成 policy / critic 观测
5. reward manager 计算速度跟踪与基础正则项
6. termination manager 判断 episode 是否结束

后续你要做 PASIST 时，建议先用这个环境确认 Isaac Lab / Go2 资产能正常工作，
再写 BasePasistEnv 适配器，不要一开始把 SIL / DTW / 多技能 target pose 全塞进来。
"""


TASK_ID = "PASIST-Go2-Minimal-Velocity"


@configclass
class MiniGo2SceneCfg(InteractiveSceneCfg):
    """最小场景配置：平地 + Go2 + 接触传感器 + 天空光。"""

    # 平地配置。
    # 这里使用 terrain_type="plane"，避免复杂 terrain generator 带来的额外变量。
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # Go2 机器人配置。
    # "{ENV_REGEX_NS}" 是 Isaac Lab 多并行环境的命名空间占位符。
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 接触传感器。
    # 最小环境也建议保留，因为 termination / feet reward / illegal contact 经常需要它。
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # 天空光，只影响可视化，不影响训练物理逻辑。
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class MiniEventCfg:
    """最小 reset 事件配置。"""

    # reset base 位姿。
    # 只随机 yaw，x/y 固定在原点附近，便于调试最小环境是否正常。
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # reset 关节。
    # position_range=(1.0, 1.0) 表示使用 robot cfg 中的默认关节姿态；
    # velocity_range 设为 0，减少最小环境中的随机性。
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class MiniCommandsCfg:
    """最小速度命令配置。"""

    # base_velocity 是 Unitree velocity 环境的核心 command。
    # 策略需要跟踪这个 command 中的 x/y 线速度和 yaw 角速度。
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.2, 0.2),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-0.5, 0.5),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.4, 0.4),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class MiniActionsCfg:
    """最小动作配置。"""

    # 使用关节位置控制。
    # policy 输出的 action 会乘以 scale，再加到默认关节位置上作为目标关节位置。
    joint_position = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )


@configclass
class MiniObservationsCfg:
    """最小观测配置。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """actor/policy 使用的观测。"""

        # base 角速度：帮助策略感知机体旋转状态。
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        # 重力在机体坐标系下的投影：常用于感知身体姿态。
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # 当前速度命令。
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            clip=(-100, 100),
            params={"command_name": "base_velocity"},
        )
        # 关节相对默认姿态的位置偏差。
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # 关节速度。
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            clip=(-100, 100),
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        # 上一时刻 action，有助于策略学习平滑控制。
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self) -> None:
            # 启用上面各 ObsTerm 中的 observation noise。
            self.enable_corruption = True
            # 拼接所有 observation terms，输出一个连续 policy observation 向量。
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """critic 使用的 privileged observation。"""

        # critic 可以额外看到 base 线速度，actor 不一定需要看到。
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            clip=(-100, 100),
            params={"command_name": "base_velocity"},
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class MiniRewardsCfg:
    """最小 reward 配置。"""

    # 线速度跟踪：这是 velocity task 的主奖励。
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # yaw 角速度跟踪。
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # 惩罚 z 方向速度，减少跳动。
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # 惩罚 roll/pitch 角速度，减少机体左右/前后晃动。
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # 惩罚力矩，降低能耗与不稳定控制。
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4)
    # 惩罚动作变化过快。
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    # 惩罚身体不水平。
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)


@configclass
class MiniTerminationsCfg:
    """最小 termination 配置。"""

    # 到达 episode 长度后截断。
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 身体倾角过大时终止。
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    # base 接触地面时终止，通常代表摔倒。
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class Go2MiniEnvCfg(ManagerBasedRLEnvCfg):
    """训练用最小 Go2 velocity 环境总配置。"""

    # 先用 64 个并行环境做最小验证；确认跑通后再增大到 1024/4096。
    scene: MiniGo2SceneCfg = MiniGo2SceneCfg(num_envs=64, env_spacing=2.5)

    observations: MiniObservationsCfg = MiniObservationsCfg()
    actions: MiniActionsCfg = MiniActionsCfg()
    commands: MiniCommandsCfg = MiniCommandsCfg()
    rewards: MiniRewardsCfg = MiniRewardsCfg()
    terminations: MiniTerminationsCfg = MiniTerminationsCfg()
    events: MiniEventCfg = MiniEventCfg()

    def __post_init__(self) -> None:
        """配置仿真步长、控制频率和传感器更新周期。"""
        # policy action 每 4 个 physics step 生效一次。
        self.decimation = 4
        # 单个 episode 最长 20 秒。
        self.episode_length_s = 20.0

        # Isaac Sim 物理步长；配合 decimation=4，控制频率约 50Hz。
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # 接触传感器每个 physics step 更新一次。
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class Go2MiniPlayEnvCfg(Go2MiniEnvCfg):
    """play / 可视化调试用最小环境配置。"""

    def __post_init__(self) -> None:
        super().__post_init__()
        # play 模式减少并行环境数，方便看单个或少量机器人行为。
        self.scene.num_envs = 16
        # play 模式使用更大的速度范围，便于测试策略泛化。
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


def register_go2_mini_env() -> None:
    """
    注册最小环境到 gymnasium。

    训练脚本可以通过 TASK_ID = "PASIST-Go2-Minimal-Velocity" 找到该环境。
    如果任务已经注册，重复调用会直接跳过，便于交互式调试。
    """
    if TASK_ID in gym.registry:
        return

    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}:Go2MiniEnvCfg",
            "play_env_cfg_entry_point": f"{__name__}:Go2MiniPlayEnvCfg",
            "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
        },
    )


# import 本文件时自动注册环境。
register_go2_mini_env()
