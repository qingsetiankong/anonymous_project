import math

# 这个文件是 Isaac Lab 的“配置式环境”定义文件。
# 它不直接写 step/reset 的运行逻辑，而是用一组 configclass 告诉 Isaac Lab：
# 1. 场景里有什么：地形、机器人、传感器、光照
# 2. 每个 episode 如何 reset、是否做 domain randomization
# 3. policy 接收哪些观测、输出什么动作
# 4. 用哪些 reward 和 termination 组成 velocity tracking 任务
# 5. 训练环境和 play/可视化环境分别使用多少并行环境、什么地形规模
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

# 这里定义的是一个“程序生成地形”的配置。
# Isaac Lab 会根据 TerrainGeneratorCfg 生成许多小地形块，并把它们铺成一个训练地形网格。
# 当前真正启用的只有 flat 平地；其它 rough / slope / stair 地形先保留为注释，
# 后续如果要做更强的鲁棒性训练或 curriculum，可以逐步打开。
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    # 单个地形子块的物理尺寸，单位通常是米。
    size=(8.0, 8.0),
    # 地形网格外侧留出的边界宽度，给机器人初始化和防止越界留安全区域。
    border_width=20.0,
    # 地形网格行数；配合 num_cols 决定一共有多少个 terrain tiles。
    num_rows=10,
    # 地形网格列数。
    num_cols=20,
    # 高度场/网格在水平面上的采样分辨率。
    horizontal_scale=0.1,
    # 高度方向缩放，值越小地形起伏越细。
    vertical_scale=0.005,
    # 坡度阈值，超过这个阈值的局部坡面会按 terrain generator 的规则处理。
    slope_threshold=0.75,
    # 地形难度范围；开启 curriculum 时通常会从低难度逐渐增加。
    difficulty_range=(0.0, 1.0),
    # 是否缓存生成结果。调试地形配置时通常关掉，稳定训练时可以考虑打开。
    use_cache=False,
    # 子地形类型及占比。当前只启用了 flat，所以环境本质上仍是平地训练。
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        # ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        # ),
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
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
class RobotSceneCfg(InteractiveSceneCfg):
    """定义 Isaac Lab 交互场景：地形、机器人、传感器和光照。"""

    # 地面/地形导入配置。
    # 这里使用 terrain generator，而不是固定 plane，因此后续可以很自然地扩展到随机地形和 curriculum。
    terrain = TerrainImporterCfg(
        # 地形在 USD stage 中的路径。
        prim_path="/World/ground",
        # "generator" 表示由 terrain_generator 程序生成；如果改成 "plane" 就是普通无限平面。
        terrain_type="generator",  # "plane", "generator"
        # 使用上面定义的 COBBLESTONE_ROAD_CFG 作为地形生成器。
        terrain_generator=COBBLESTONE_ROAD_CFG,  # None, ROUGH_TERRAINS_CFG
        # 每个环境初始化时允许采样到的最高地形难度等级。
        max_init_terrain_level=1,
        # -1 表示使用默认碰撞组设置，通常可以和所有物体发生碰撞。
        collision_group=-1,
        # 地形物理材质：摩擦、反弹等。velocity tracking 对摩擦很敏感。
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        # 地形视觉材质，只影响渲染效果，不影响物理训练。
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        # 是否显示地形调试可视化。
        debug_vis=False,
    )
    # 机器人配置。
    # ROBOT_CFG 来自 unitree_rl_lab，replace 只修改 prim_path，
    # "{ENV_REGEX_NS}" 是 Isaac Lab 多环境复制时使用的命名空间占位符。
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 高度扫描传感器。
    # 它从机器人 base 上方发射一组 ray，测量局部地形高度。
    # 当前 policy 观测里没有启用 height_scan，但后续做复杂地形时可以加进 observation。
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        # 从 base 上方 20m 的位置向下扫描，避免射线起点落在机器人或地面内部。
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        # ray pattern 随机器人 yaw 旋转，保持相对机器人朝向的局部地形感知。
        ray_alignment="yaw",
        # 扫描网格分辨率和范围：resolution 越小，地形信息越密，计算也越贵。
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        # 射线只与地面 mesh 求交。
        mesh_prim_paths=["/World/ground"],
    )
    # 接触传感器。
    # 用来检测脚掌是否接触地面、身体是否发生非法碰撞，也支持 feet_air_time 等 reward。
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # 天空光源，只影响可视化。
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """定义环境事件：启动随机化、reset 随机化、周期扰动。"""

    # startup 事件只在环境启动时执行一次，常用于 domain randomization 的静态属性随机化。
    # 这里随机化机器人各刚体的摩擦和恢复系数，提升 sim-to-real 和不同地面条件鲁棒性。
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )

    # 启动时随机给 base 增加质量。
    # 这等价于模拟电池、载荷、模型质量误差等不确定性。
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset 事件在每次 episode reset 时执行。
    # 当前 force/torque 范围是 0，相当于预留接口但不施加初始外力。
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    # reset 机器人根节点状态。
    # pose_range 控制初始 x/y/yaw 随机化；velocity_range 目前全 0，表示初始速度清零。
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
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

    # reset 关节状态。
    # position_range=(1.0, 1.0) 通常表示按默认关节位置等比例初始化；
    # velocity_range 给关节初速度加随机扰动，增强鲁棒性。
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval 事件会在 episode 运行过程中周期性触发。
    # 这里每隔 5~10 秒给机器人 base 设置一次随机速度，用来模拟外部推搡扰动。
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """定义任务命令，也就是策略需要跟踪的目标速度。"""

    # base_velocity 是速度跟踪任务的核心 command。
    # 它会周期性采样线速度 x/y 和角速度 z，policy 的目标就是让机器人跟踪这些命令。
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        # command 作用到哪个资产上，这里是 RobotSceneCfg 中命名为 "robot" 的 Articulation。
        asset_name="robot",
        # 每隔 10 秒重新采样一次速度命令。
        resampling_time_range=(10.0, 10.0),
        # 10% 环境采样“站立/零速度”命令，有助于学会静止稳定。
        rel_standing_envs=0.1,
        # 是否显示 command 可视化箭头，调试时很有用，训练大规模环境时可关闭。
        debug_vis=True,
        # 初始 command 采样范围。这里 x/y 比较小，角速度范围较大。
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-1, 1)
        ),
        # command curriculum 或 play 模式可使用的最大范围。
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class ActionsCfg:
    """定义策略输出动作如何映射到机器人控制命令。"""

    # 这里使用关节位置控制。
    # policy 输出会乘以 scale=0.25，然后叠加默认关节 offset，
    # 最终作为各关节的位置目标发送给 Isaac Lab action manager。
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )


@configclass
class ObservationsCfg:
    """定义 actor/critic 可以看到哪些观测量。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """policy/actor 使用的观测组。"""

        # 观测项的定义顺序会影响 concatenate 后的向量顺序，后续做 imitation_obs 切片时要注意。
        # base_ang_vel: base 角速度，scale=0.2 用于缩放数值，noise 用于观测噪声随机化。
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        # projected_gravity: 重力向量投影到机体坐标系，常用于估计身体姿态。
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        # 当前速度命令，即 CommandsCfg.base_velocity 采样得到的目标速度。
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        # 关节相对默认姿态的位置偏差。
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        # 关节相对速度，scale=0.05 降低速度数值量级，noise 模拟传感器误差。
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        # 上一时刻动作，帮助策略学习动作平滑和动态控制。
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            # self.history_length = 5
            # 开启观测噪声/扰动。上面各 ObsTerm 中的 noise 只有在这里打开后才会生效。
            self.enable_corruption = True
            # 把上面各 observation term 拼成一个连续向量，方便 policy 网络直接输入。
            self.concatenate_terms = True

    # actor/policy 使用的观测组实例。
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """critic 使用的 privileged observation 观测组。"""

        # critic 通常可以看到比 actor 更多的信息，训练时用于更稳定的 value 估计；
        # 部署到真实机器人时通常只保留 actor，因此不要让 actor 依赖这些 privileged 量。
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        # height_scanner = ObsTerm(func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 5.0),
        # )

        # def __post_init__(self):
        #     self.history_length = 5

    # privileged observations：只给 critic 使用。
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """定义 Isaac Lab 原生 velocity tracking 任务的 reward terms。"""

    # -- task tracking rewards -------------------------------------------------
    # 线速度 x/y 跟踪奖励，鼓励机器人 base 速度匹配 base_velocity command。
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # yaw 角速度跟踪奖励，鼓励机器人按命令转向。
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- base / joint regularization rewards ----------------------------------
    # 惩罚 z 方向线速度，避免机器人上下跳动。
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # 惩罚 roll/pitch 角速度，减少身体左右/前后剧烈晃动。
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # 惩罚关节速度，促进更平滑、更省能的步态。
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    # 惩罚关节加速度，降低动作抖动。
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # 惩罚关节力矩，降低能耗和硬件冲击。
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)
    # 惩罚连续动作变化率，是 locomotion 中非常常见的平滑项。
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    # 惩罚关节位置接近或超过限制，保护机器人关节。
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    # 能耗惩罚，通常与 torque * velocity 等量有关。
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    # -- robot posture rewards -------------------------------------------------
    # 惩罚身体不水平，鼓励 base 保持较平的姿态。
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)

    # 惩罚关节偏离默认姿态。
    # 当命令速度很小时 stand_still_scale 会放大惩罚，帮助机器人学会静止站稳。
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )

    # -- feet contact / gait rewards ------------------------------------------
    # 奖励脚的腾空时间，帮助形成更自然的步态节律。
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    # 惩罚各脚腾空时间方差过大，鼓励步态节律更均衡。
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    # 惩罚脚掌接触地面时滑动，减少打滑和不稳定。
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    # feet_contact_forces = RewTerm(
    #     func=mdp.contact_forces,
    #     weight=-0.02,
    #     params={
    #         "threshold": 100.0,
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #     },
    # )

    # -- safety / illegal contact rewards -------------------------------------
    # 惩罚非期望部位接触地面，例如头、髋、大腿、小腿。
    # 对四足 locomotion 来说，这能减少拖腿、摔倒、身体碰地等行为。
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]),
        },
    )


@configclass
class TerminationsCfg:
    """定义 episode 结束条件。"""

    # 时间达到 episode_length_s 时结束；time_out=True 表示这是时间截断而非失败。
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base 接触地面时终止，通常代表摔倒或身体碰撞。
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    # 身体倾角超过阈值时终止，防止策略在严重翻倒状态继续采样无意义数据。
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """定义课程学习项。"""

    # 根据机器人在地形上的表现调整地形等级。
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # 根据训练进度或表现扩大线速度 command 范围。
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """训练用 Go2 velocity-tracking 环境总配置。"""

    # 场景设置：4096 个并行环境，环境间距 2.5m。
    # 并行环境越多，采样效率越高，但显存/计算需求也越大。
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # 基础 MDP 模块：观测、动作、命令。
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP 训练模块：奖励、终止条件、随机事件、课程学习。
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """配置对象创建后的二次初始化。"""
        # decimation 表示 policy action 每隔多少个 physics step 执行一次。
        # 这里 physics dt=0.005，decimation=4，所以控制频率约为 1 / (0.005*4) = 50Hz。
        self.decimation = 4
        # 单个 episode 的最大仿真时长，单位秒。
        self.episode_length_s = 20.0
        # 物理仿真的基础步长，单位秒。
        self.sim.dt = 0.005
        # 渲染间隔与 action decimation 保持一致，避免每个 physics step 都渲染。
        self.sim.render_interval = self.decimation
        # 使用 terrain 的物理材质作为全局 sim 物理材质。
        self.sim.physics_material = self.scene.terrain.physics_material
        # PhysX GPU 刚体接触 patch 上限；地形/并行环境很多时需要调大，避免接触计算资源不足。
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # 传感器更新周期。
        # contact sensor 每个 physics step 更新；height scanner 按 policy 控制周期更新，降低开销。
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # 如果启用了 terrain_levels curriculum，就让 terrain generator 根据课程学习调整难度。
        # 如果 curriculum 中没有 terrain_levels，则关闭地形课程学习，使用固定难度地形。
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """play / 可视化调试用环境配置。"""

    def __post_init__(self):
        super().__post_init__()
        # play 模式不需要 4096 个环境，减少到 32 个方便可视化和快速调试。
        self.scene.num_envs = 32
        # 地形网格也缩小，减少加载和渲染开销。
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        # play 模式直接使用最大 command 范围，方便测试策略在完整速度范围内的表现。
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
