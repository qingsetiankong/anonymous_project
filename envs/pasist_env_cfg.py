from __future__ import annotations

import math
import pathlib

import gymnasium as gym
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import numpy as np
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from go2_joint_order import GO2_JOINT_ORDER
from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp


"""
PASIST 专用 Isaac Lab 环境配置。

这个文件的定位介于：
- `envs/isaacsim_mini_envs.py` 的“最小验证环境”
- 真正完整的多技能 PASIST 环境

之间。

设计目标：
1. 先提供一个能稳定运行的 PASIST 基础环境配置
2. 保持和你当前 `IsaacLabPasistEnv` 适配器一致
3. 给未来扩展 skill、target pose、更多传感器和 reward 留接口

当前实现策略：
- 仍以 Go2 velocity locomotion 为底层任务
- 只实现单技能 `walk` 的基础版本
- skill 相关元数据不直接塞进 Isaac Lab 的 command/reward manager，
  而是通过 `pasist` 配置块暴露给上层 wrapper / trainer 使用

这样做的好处是：
- 底层 Isaac Lab 结构保持稳定
- 上层 PASIST 逻辑可以独立演进
- 后续扩成多技能时，不需要推翻整个环境配置
"""


TASK_ID = "PASIST-Go2-Base"
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
INIT_POSE_PATH = PROJECT_ROOT / "init_pose" / "init_pose.npy"

def _load_init_joint_positions(init_pose_path: pathlib.Path = INIT_POSE_PATH) -> dict[str, float]:
    """
    从 `init_pose.npy` 加载 12 关节初始位置，并转换成 Isaac Lab 可用的关节字典。

    文件要求：
    - 支持 shape = `(12,)` 或 `(1, 12)`
    - 数据顺序必须与 `GO2_JOINT_ORDER` 一致

    返回：
    - `dict[str, float]`
      例如：
      {
        "FR_hip_joint": ...,
        "FR_thigh_joint": ...,
        ...
      }
    """
    if not init_pose_path.exists():
        raise FileNotFoundError(f"找不到初始姿态文件: {init_pose_path}")

    init_pose = np.load(init_pose_path).astype(np.float32).reshape(-1)
    if init_pose.shape[0] != len(GO2_JOINT_ORDER):
        raise ValueError(
            f"初始姿态维度不正确，期望 {len(GO2_JOINT_ORDER)} 个关节值，实际得到 {init_pose.shape[0]}"
        )

    return {joint_name: float(joint_value) for joint_name, joint_value in zip(GO2_JOINT_ORDER, init_pose)}


GO2_INIT_JOINT_POS = _load_init_joint_positions()
GO2_PASIST_ROBOT_CFG = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
GO2_PASIST_ROBOT_CFG.init_state.joint_pos = GO2_INIT_JOINT_POS
GO2_PASIST_ROBOT_CFG.init_state.joint_vel = {".*": 0.0}

# PASIST 地形生成器配置。
# 当前默认仍然只启用 flat，使训练行为继续接近平地；
# 但结构上已经和 velocity_env_cfg.py 对齐，后续可以直接打开 rough/slope/stairs。
PASIST_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.2,
        #     noise_range=(0.01, 0.06),
        #     noise_step=0.01,
        #     border_width=0.25,
        # ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.2,
        #     slope_range=(0.0, 0.4),
        #     platform_width=2.0,
        #     border_width=0.25,
        # ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2,
        #     grid_width=0.45,
        #     grid_height_range=(0.05, 0.2),
        #     platform_width=2.0,
        # ),
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
    },
)


@configclass
class PasistMetadataCfg:
    """
    PASIST 任务的附加元数据配置。

    这一部分不是 Isaac Lab manager 的标准字段，
    而是给项目上层 wrapper / trainer 读取的。

    当前你最需要的几项都放在这里：
    - skill 名称
    - 默认 skill
    - command manager 中 velocity command 的名字
    - actor / critic observation 的键名
    - imitation observation 的默认切片

    后续扩展多技能时，这个配置类会非常有用。
    """

    # 当前版本只先支持一个技能，但结构上已经支持未来扩展到多个技能。
    skill_names: tuple[str, ...] = ("walk",)
    # 默认训练/采样时使用的技能编号。
    default_skill_id: int = 0

    # 外层 `IsaacLabPasistEnv` 用这个名字去 command manager 中取 velocity command。
    command_name: str = "base_velocity"
    # Isaac Lab observation dict 中 actor / critic 的键名。
    policy_obs_key: str = "policy"
    critic_obs_key: str = "critic"

    # imitation observation 默认从 policy observation 中切出：
    # 0:6   -> base_ang_vel + projected_gravity
    # 9:33  -> joint_pos_rel + joint_vel_rel
    # 这样跳过 velocity command 和 last_action，减少无关噪声。
    imitation_slices: tuple[tuple[int, int], ...] = ((0, 6), (9, 33))

    # target pose 初始化策略。
    # 当前推荐使用 `reset_observation`：
    # 第一次 reset 后，以上层 wrapper 提取出的 imitation observation 作为初始 target pose。
    target_pose_source: str = "reset_observation"

    # 这个字段用来提醒后续实现者：skill command 当前由外层 wrapper 注入，
    # 不是 Isaac Lab command manager 原生多技能管理。
    skill_command_source: str = "external_wrapper"


@configclass
class PasistGo2SceneCfg(InteractiveSceneCfg):
    """
    PASIST 基础场景配置。

    当前地形实现已经升级成“generator 骨架 + flat 默认子地形”。
    这样你既能保持当前几乎等价于平地的训练行为，又能在后续较平滑地切换到复杂地形。

    后面如果你要提升鲁棒性，可以优先在这里扩展：
    - 在 `PASIST_TERRAIN_CFG.sub_terrains` 中打开 rough / slope / stairs
    - 把 `height_scanner` 观测接进 policy 或 critic
    - 加入更多传感器
    """

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=PASIST_TERRAIN_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # 这里显式使用本项目的 `init_pose/init_pose.npy` 作为默认关节初始位置。
    # 又因为 `PasistEventCfg.reset_robot_joints` 使用的是 `reset_joints_by_scale`
    # 且 position_range=(1.0, 1.0)，所以每个 episode reset 时都会回到这套关节姿态。
    robot: ArticulationCfg = GO2_PASIST_ROBOT_CFG

    # 为未来复杂地形感知预留高度扫描传感器。
    # 当前它还没有接进 observation，因此不会改变现有训练输入维度。
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class PasistEventCfg:
    """
    PASIST 基础事件配置。

    当前保持非常克制的随机化：
    - reset root state
    - reset joints

    这是为了先降低环境非平稳性，帮助你调通：
    - target pose
    - imitation observation
    - trajectory selector
    - SIL buffer

    后续再逐步加：
    - 质量随机化
    - 摩擦随机化
    - 推搡扰动
    """

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

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class PasistCommandsCfg:
    """
    PASIST 基础 command 配置。

    当前阶段只保留一个 velocity command：
    - `base_velocity`

    重要说明：
    - 这个 command 是 Isaac Lab 层的底层速度命令
    - skill 选择仍由上层 `IsaacLabPasistEnv` 和 `SkillSelector` 管理

    后续如果你真的想把 skill 也塞进 Isaac Lab command manager，
    再在这里引入新的 command term 会更合适。
    """

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        # 故意设置得很长，减少底层自行重采样带来的干扰。
        # 真实训练时外层 wrapper 会继续显式覆写 command。
        resampling_time_range=(1.0e6, 1.0e6),
        rel_standing_envs=0.0,
        debug_vis=False,
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
class PasistActionsCfg:
    """
    PASIST 基础动作配置。

    当前仍然使用关节位置控制，这是最容易和已有 Go2 locomotion 任务对齐的做法。
    """

    joint_position = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )


@configclass
class PasistObservationsCfg:
    """
    PASIST 基础观测配置。

    当前策略：
    - IsaacLab 原始 policy observation 保持和最小环境一致，方便复用现有 wrapper 切片逻辑
    - 真正喂给 actor 的输入会在 wrapper 里额外拼接 4 维 skill one-hot command
    - critic observation 额外保留 base_lin_vel，方便上层提 measured velocity

    未来扩展方向：
    - 引入 height_scan
    - 拆分 imitation-friendly observation group
    - 显式加入 skill one-hot / target pose embedding
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """actor / policy 使用的观测组。"""

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            clip=(-100.0, 100.0),
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            clip=(-100.0, 100.0),
            params={"command_name": "base_velocity"},
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            clip=(-100.0, 100.0),
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            clip=(-100.0, 100.0),
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """critic 使用的 privileged observation。"""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100.0, 100.0))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            clip=(-100.0, 100.0),
            params={"command_name": "base_velocity"},
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100.0, 100.0))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100.0, 100.0))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class PasistRewardsCfg:
    """
    PASIST 基础 reward 配置。

    这里的 reward 不是论文最终的 PASIST 总奖励。
    它的职责更像：
    - 给底层 Isaac Lab locomotion 提供一个稳定的原生行为先验
    - 在你还没完全接管 trainer 时，环境本身依然能工作

    真正的 PASIST 总奖励组合：
    - `task reward`
    - `sil reward`
    - `regularization reward`

    建议继续放在你自己的 `rewards/` 模块和 trainer 中做，而不是全部塞进这里。
    """

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)


@configclass
class PasistTerminationsCfg:
    """
    PASIST 基础终止条件配置。

    只保留最核心的三项：
    - 超时
    - 身体倾斜过大
    - base 接触地面
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class Go2PasistEnvCfg(ManagerBasedRLEnvCfg):
    """
    Go2 的 PASIST 基础环境总配置。

    这个类是你后面最应该传给 `gym.make(..., cfg=...)` 的主配置类之一。
    它除了标准 Isaac Lab 字段外，还包含一个 `pasist` 配置块，
    供你的 wrapper / trainer 读取。
    """

    scene: PasistGo2SceneCfg = PasistGo2SceneCfg(num_envs=64, env_spacing=2.5)
    observations: PasistObservationsCfg = PasistObservationsCfg()
    actions: PasistActionsCfg = PasistActionsCfg()
    commands: PasistCommandsCfg = PasistCommandsCfg()
    rewards: PasistRewardsCfg = PasistRewardsCfg()
    terminations: PasistTerminationsCfg = PasistTerminationsCfg()
    events: PasistEventCfg = PasistEventCfg()

    # 这个字段不会被 Isaac Lab manager 自动消费，
    # 但会被项目自己的 wrapper / trainer 用到。
    pasist: PasistMetadataCfg = PasistMetadataCfg()

    def __post_init__(self) -> None:
        """
        配置仿真步长、控制频率和传感器更新周期。
        """
        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class Go2PasistPlayEnvCfg(Go2PasistEnvCfg):
    """
    可视化 / play 模式下的 PASIST 基础环境配置。
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


def register_go2_pasist_env() -> None:
    """
    注册 PASIST 基础环境到 Gymnasium。

    注册后可以通过：
    `gym.make("PASIST-Go2-Base", cfg=Go2PasistEnvCfg())`
    创建环境。
    """
    if TASK_ID in gym.registry:
        return

    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}:Go2PasistEnvCfg",
            "play_env_cfg_entry_point": f"{__name__}:Go2PasistPlayEnvCfg",
            "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
        },
    )


# import 本文件时自动注册。
register_go2_pasist_env()
