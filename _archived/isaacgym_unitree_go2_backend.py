from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

from go2_joint_order import GO2_JOINT_ORDER, build_reorder_indices

class UnitreeGo2PasistBackend:
    """
    unitree_rl_gym Go2 task 到 PASIST IsaacGymPasistEnv 的轻量后端包装器。

    这个类不替代 Unitree 的 `LeggedRobot`，而是复用它负责 Isaac Gym 仿真、
    Go2 URDF 加载、PD 控制、reset 和接触终止；同时把观测重排为当前 PASIST
    训练器已经约定好的格式：

    policy obs, 45 维:
      [base_ang_vel*0.2, projected_gravity, command_xyz,
       joint_pos_rel, joint_vel*0.05, last_action]

    critic states, 48 维:
      [base_lin_vel, base_ang_vel*0.2, projected_gravity, command_xyz,
       joint_pos_rel, joint_vel*0.05, last_action]
    """

    policy_obs_dim = 45
    critic_obs_dim = 48

    def __init__(
        self,
        task_name: str = "go2",
        num_envs: int = 4096,
        seed: int = 0,
        sim_device: str = "cuda:0",
        headless: bool = True,
        use_gpu_pipeline: bool = True,
        use_gpu: bool | None = None,
        subscenes: int = 0,
        num_threads: int = 10,
        disable_noise: bool = True,
        disable_domain_rand: bool = True,
        command_resampling_time: float = 1.0e9,
        randomize_episode_phase: bool = True,
    ) -> None:
        self.task_name = str(task_name)
        self._requested_num_envs = int(num_envs)
        self._requested_seed = int(seed)
        self._requested_sim_device = str(sim_device)
        self.headless = bool(headless)
        self.randomize_episode_phase = bool(randomize_episode_phase)
        self._pasist_command_velocity = 0.0

        self._env, self.cfg = self._make_unitree_env(
            task_name=self.task_name,
            num_envs=self._requested_num_envs,
            seed=self._requested_seed,
            sim_device=self._requested_sim_device,
            headless=self.headless,
            use_gpu_pipeline=bool(use_gpu_pipeline),
            use_gpu=use_gpu,
            subscenes=int(subscenes),
            num_threads=int(num_threads),
            disable_noise=bool(disable_noise),
            disable_domain_rand=bool(disable_domain_rand),
            command_resampling_time=float(command_resampling_time),
        )

        self.num_envs = int(self._env.num_envs)
        self.num_actions = int(self._env.num_actions)
        self.num_obs = self.policy_obs_dim
        self.num_states = self.critic_obs_dim
        self.device = str(self._env.device)
        self.policy_joint_order = tuple(GO2_JOINT_ORDER)
        self.asset_joint_order = self._resolve_asset_joint_order()
        self._asset_to_policy_indices = build_reorder_indices(
            self.asset_joint_order,
            self.policy_joint_order,
        )
        self._policy_to_asset_indices = build_reorder_indices(
            self.policy_joint_order,
            self.asset_joint_order,
        )

        self.obs_buf = None
        self.states_buf = None
        self._refresh_pasist_buffers()

    @property
    def root_states(self):
        return self._env.root_states

    @property
    def dof_pos(self):
        return self._reorder_asset_to_policy(self._env.dof_pos)

    @property
    def dof_vel(self):
        return self._reorder_asset_to_policy(self._env.dof_vel)

    @property
    def default_dof_pos(self):
        return self._reorder_asset_to_policy(self._env.default_dof_pos)

    @property
    def commands(self):
        return self._env.commands

    @property
    def inner_env(self):
        return self._env

    @property
    def step_dt(self) -> float:
        return float(getattr(self._env, "dt", 0.02))

    def reset(self):
        self._apply_pasist_command()
        self._env.reset()
        self._randomize_episode_phase_after_full_reset()
        self._apply_pasist_command()
        self._refresh_pasist_buffers()
        return {"obs": self.obs_buf, "states": self.states_buf}

    def step(self, actions):
        import torch

        policy_action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=self._env.device)
        if policy_action_tensor.ndim == 1:
            policy_action_tensor = policy_action_tensor.reshape(1, self.num_actions)
        if policy_action_tensor.shape != (self.num_envs, self.num_actions):
            raise ValueError(
                "actions shape 不匹配，"
                f"期望 {(self.num_envs, self.num_actions)}，实际为 {tuple(policy_action_tensor.shape)}"
            )
        action_tensor = self._reorder_policy_to_asset(policy_action_tensor)

        self._apply_pasist_command()
        _, _, reward, done, extras = self._env.step(action_tensor)

        # Unitree task 会在 reset/post-step 内部重采样 command；这里把 PASIST command 写回去。
        self._apply_pasist_command()
        self._refresh_pasist_buffers()

        info = dict(extras or {})
        if hasattr(self._env, "time_out_buf"):
            info["time_outs"] = self._env.time_out_buf
        return {"obs": self.obs_buf, "states": self.states_buf}, reward, done, info

    def set_pasist_command(self, command: Any) -> None:
        self._pasist_command_velocity = float(getattr(command, "velocity", command))
        self._apply_pasist_command()

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        gym = getattr(self._env, "gym", None)
        sim = getattr(self._env, "sim", None)
        viewer = getattr(self._env, "viewer", None)
        if gym is None:
            return
        if viewer is not None:
            try:
                gym.destroy_viewer(viewer)
            except Exception:
                pass
        if sim is not None:
            try:
                gym.destroy_sim(sim)
            except Exception:
                pass

    def _apply_pasist_command(self) -> None:
        commands = getattr(self._env, "commands", None)
        if commands is None:
            return
        commands[:, 0] = float(self._pasist_command_velocity)
        commands[:, 1] = 0.0
        commands[:, 2] = 0.0
        if commands.shape[1] > 3:
            commands[:, 3] = 0.0

    def _refresh_pasist_buffers(self) -> None:
        import torch

        env = self._env
        command_xyz = env.commands[:, :3]
        joint_pos_rel = self.dof_pos - self.default_dof_pos
        joint_vel_scaled = self.dof_vel * 0.05
        base_ang_vel_scaled = env.base_ang_vel * 0.2
        last_action = self._reorder_asset_to_policy(env.actions)

        self.obs_buf = torch.cat(
            (
                base_ang_vel_scaled,
                env.projected_gravity,
                command_xyz,
                joint_pos_rel,
                joint_vel_scaled,
                last_action,
            ),
            dim=-1,
        )
        self.states_buf = torch.cat(
            (
                env.base_lin_vel,
                base_ang_vel_scaled,
                env.projected_gravity,
                command_xyz,
                joint_pos_rel,
                joint_vel_scaled,
                last_action,
            ),
            dim=-1,
        )

    def _randomize_episode_phase_after_full_reset(self) -> None:
        """
        在全量 reset 后打散 episode 计数，避免所有环境在同一时刻触发超时。

        说明:
        - unitree_rl_gym 的 `episode_length_buf` 会在每个 physics step 自增
        - timeout 判定依赖它是否超过 `max_episode_length`
        - 如果所有 env 在训练开始时同步清零，后续就会几乎同步撞到 timeout
        - 这里仅在“全量 reset”后随机化一次相位，不修改单个 env 的常规 reset 逻辑
        """
        if not self.randomize_episode_phase:
            return

        import torch

        episode_length_buf = getattr(self._env, "episode_length_buf", None)
        max_episode_length = getattr(self._env, "max_episode_length", None)
        if episode_length_buf is None or max_episode_length is None:
            return

        try:
            max_episode_steps = int(max_episode_length)
        except (TypeError, ValueError):
            return

        if max_episode_steps <= 1:
            return

        randomized_phase = torch.randint(
            low=0,
            high=max_episode_steps,
            size=episode_length_buf.shape,
            device=episode_length_buf.device,
            dtype=episode_length_buf.dtype,
        )
        episode_length_buf.copy_(randomized_phase)

        time_out_buf = getattr(self._env, "time_out_buf", None)
        if time_out_buf is not None:
            time_out_buf.zero_()

    def _resolve_asset_joint_order(self) -> tuple[str, ...]:
        dof_names = tuple(str(name) for name in getattr(self._env, "dof_names", ()))
        if len(dof_names) != self.num_actions:
            raise ValueError(
                "无法从 unitree_rl_gym backend 推断完整 dof 顺序："
                f" num_actions={self.num_actions}, dof_names={dof_names}"
            )
        return dof_names

    def _reorder_asset_to_policy(self, value):
        return self._reorder_last_dim(value, self._asset_to_policy_indices)

    def _reorder_policy_to_asset(self, value):
        return self._reorder_last_dim(value, self._policy_to_asset_indices)

    def _reorder_last_dim(self, value, indices: tuple[int, ...]):
        if value is None:
            return None
        return value[..., list(indices)]

    def _make_unitree_env(
        self,
        task_name: str,
        num_envs: int,
        seed: int,
        sim_device: str,
        headless: bool,
        use_gpu_pipeline: bool,
        use_gpu: bool | None,
        subscenes: int,
        num_threads: int,
        disable_noise: bool,
        disable_domain_rand: bool,
        command_resampling_time: float,
    ):
        from numpy_compat import ensure_numpy_legacy_aliases

        ensure_numpy_legacy_aliases()
        from isaacgym import gymapi

        import legged_gym.envs  # noqa: F401 - 注册 task_registry 中的 go2
        from legged_gym.utils.helpers import class_to_dict, parse_sim_params, set_seed
        from legged_gym.utils.task_registry import task_registry

        env_cfg, _ = task_registry.get_cfgs(task_name)
        env_cfg = copy.deepcopy(env_cfg)
        env_cfg.seed = int(seed)
        env_cfg.env.num_envs = int(num_envs)
        env_cfg.commands.heading_command = False
        env_cfg.commands.resampling_time = float(command_resampling_time)
        env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
        env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
        env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
        env_cfg.commands.ranges.heading = [0.0, 0.0]
        if disable_noise:
            env_cfg.noise.add_noise = False
        if disable_domain_rand:
            env_cfg.domain_rand.randomize_friction = False
            env_cfg.domain_rand.randomize_base_mass = False
            env_cfg.domain_rand.push_robots = False

        if use_gpu is None:
            use_gpu = str(sim_device).startswith("cuda")

        args = SimpleNamespace(
            task=task_name,
            num_envs=int(num_envs),
            seed=int(seed),
            max_iterations=None,
            resume=False,
            experiment_name=None,
            run_name=None,
            load_run=None,
            checkpoint=None,
            headless=bool(headless),
            horovod=False,
            rl_device=sim_device,
            sim_device=sim_device,
            device=sim_device,
            physics_engine=gymapi.SIM_PHYSX,
            use_gpu=bool(use_gpu),
            use_gpu_pipeline=bool(use_gpu_pipeline),
            subscenes=int(subscenes),
            num_threads=int(num_threads),
        )

        set_seed(env_cfg.seed)
        sim_params = parse_sim_params(args, {"sim": class_to_dict(env_cfg.sim)})
        task_class = task_registry.get_task_class(task_name)
        env = task_class(
            cfg=env_cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            sim_device=args.sim_device,
            headless=args.headless,
        )
        return env, env_cfg


def make_unitree_go2_pasist_backend(**kwargs) -> UnitreeGo2PasistBackend:
    return UnitreeGo2PasistBackend(**kwargs)
