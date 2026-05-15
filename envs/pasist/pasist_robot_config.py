from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class PasistRobotCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 49   # 45 proprio + 4 onehot
        num_privileged_obs = 52  # 48 proprio + 4 onehot
        num_actions = 12
        send_timeouts = True
        episode_length_s = 20

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = False

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.5
        num_commands = 3
        resampling_time = 10.0
        heading_command = False

        class ranges:
            lin_vel_x = [-0.5, 0.5]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-1.0, 1.0]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]

        default_joint_angles = {
            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "FR_thigh_joint": 0.8,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"joint": 40.0}
        damping = {"joint": 1.0}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf"
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = True
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.9
        base_height_target = 0.32

        class scales(LeggedRobotCfg.rewards.scales):
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            torques = 0.0
            dof_vel = 0.0
            dof_acc = -2.5e-7
            base_height = -10.0
            feet_air_time = 0.0
            collision = 0.0
            feet_stumble = 0.0
            action_rate = -0.01
            stand_still = 0.0

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1.0, 2.0]
        push_robots = True
        push_interval_s = 15.0
        max_push_vel_xy = 1.0

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05

        clip_observations = 100.0
        clip_actions = 100.0

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0

    class sim(LeggedRobotCfg.sim):
        dt = 0.005

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1


class PasistRobotCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1.0e-3
        max_grad_norm = 1.0
        num_learning_epochs = 5
        num_mini_batches = 4
        mini_batch_size = 256
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        schedule = "adaptive"

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [256, 256]
        critic_hidden_dims = [256, 256]
        activation = "elu"

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "PasistActorCritic"
        algorithm_class_name = "PasistPPO"
        num_steps_per_env = 24
        max_iterations = 5000
        save_interval = 100
        experiment_name = "pasist_phase1"
        run_name = "go2_flat"
