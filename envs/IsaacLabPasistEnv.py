from __future__ import annotations

import importlib
from typing import Any, Sequence

import gymnasium as gym
import numpy as np

from envs.BasePasistEnv import BasePasistEnv, PasistCommand


class IsaacLabPasistEnv(BasePasistEnv):
    """
    Isaac Lab 环境到 PASIST 项目接口的适配器。

    这个类的定位很重要：
    - 底层仍然使用 Isaac Lab / Isaac Sim 的原生 Gym 环境
    - 上层 trainer / SIL / rewards 只和 BasePasistEnv 交互

    当前版本的目标是：
    1. 先支持一个技能的可用版本，保证 reset / step / command / target pose 可跑通
    2. 提前把多技能扩展点留出来，后续接入 SkillSelector 时不需要推翻重写

    当前实现的设计取舍：
    - 默认包装 `PASIST-Go2-Minimal-Velocity`
    - 默认所有并行环境共享同一个 skill command
    - 默认 imitation observation 从 policy observation 的子区间中提取
    - 默认 target pose 如果没有显式提供，就在第一次 reset 时用当前机器人姿态初始化

    注意：
    - 使用本类前，外部脚本应当先通过 `AppLauncher` 启动 Isaac Sim
    - 本类本身不负责启动仿真 App，只负责环境适配
    """

    def __init__(
        self,
        task_id: str | None = None,
        env_cfg: Any | None = None,
        num_envs: int = 1,
        skill_names: Sequence[str] | None = None,
        target_pose_bank: Sequence[np.ndarray] | dict[int, np.ndarray] | None = None,
        skill_one_hot_map: dict[int, np.ndarray] | None = None,
        default_skill_id: int = 0,
        velocity_range: tuple[float, float] = (-0.5, 0.5),
        command_name: str = "base_velocity",
        policy_obs_key: str = "policy",
        critic_obs_key: str = "critic",
        imitation_slices: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        """
        参数说明：
        - task_id:
          要包装的 Gym task id。默认使用最小 Go2 velocity 环境。
        - env_cfg:
          可选的 Isaac Lab cfg；不传则自动从 task 注册信息实例化。
        - num_envs:
          并行环境数量。当前版本支持 vectorized env，但默认所有 env 共用同一 command。
        - skill_names:
          技能名称列表。当前先支持单技能，但结构上允许未来扩展到多技能。
        - target_pose_bank:
          每个 skill 对应的 target pose。
          如果不传，则第一次 reset 后自动用当前 imitation observation 初始化。
        - skill_one_hot_map:
          skill_id 到 one-hot 编码的映射。
          如果提供，则 actor 输入中的技能 command 和 `PasistCommand.one_hot`
          都以这里为准；如果不提供，则回退到标准单位 one-hot。
        - default_skill_id:
          默认技能编号。
        - velocity_range:
          `sample_command()` 的速度采样范围。
        - command_name:
          Isaac Lab 中 command manager 里 velocity command 的名字。
        - policy_obs_key / critic_obs_key:
          Isaac Lab observation dict 中 actor / critic 观测的键名。
        - imitation_slices:
          旧版用于从 policy observation 中裁剪 imitation observation 的切片。
          当前工程已经改成 pose-only imitation space：
          - base_height (1)
          - joint_pos_rel (12)
          - skill_id (1)
          因此这个参数保留兼容性，但当前默认实现不再直接使用它来构造 imitation observation。
        """
        self._register_builtin_tasks()

        if task_id is None:
            mini_envs = importlib.import_module("envs.isaacsim_mini_envs")
            task_id = mini_envs.TASK_ID

        self._task_id = str(task_id)
        self._policy_obs_key = policy_obs_key
        self._critic_obs_key = critic_obs_key
        self._command_name = command_name
        self._default_skill_id = int(default_skill_id)
        self._velocity_range = (float(velocity_range[0]), float(velocity_range[1]))

        if skill_names is None:
            skill_names = ["walk"]
        self._skill_names = [str(name) for name in skill_names]
        if not self._skill_names:
            raise ValueError("skill_names 不能为空")
        if not 0 <= self._default_skill_id < len(self._skill_names):
            raise ValueError(f"default_skill_id 超出范围: {self._default_skill_id}")

        self._skill_one_hot_map = self._normalize_skill_one_hot_map(skill_one_hot_map)
        self._command_dim = self._infer_command_dim()
        self._validate_skill_one_hot_coverage()

        # 当前 pose-only imitation space 只真正依赖 joint_pos_rel；
        # base_height 直接从 IsaacLab 机器人状态中读取；
        # 最后一维保存当前 skill_id。
        # 顺序约定为：
        # - base_height (1)
        # - joint_pos_rel (12)
        # - skill_id (1)
        # actor 输入则在原始 policy observation 的末尾额外拼接技能 one-hot command。
        self._imitation_slices = list(imitation_slices or ((0, 6), (9, 33)))
        self._joint_pos_rel_slice = (9, 21)

        self._env_cfg = self._resolve_env_cfg(task_id=self._task_id, env_cfg=env_cfg, num_envs=num_envs)
        self._freeze_command_resampling(self._env_cfg)
        self._env = gym.make(self._task_id, cfg=self._env_cfg)
        self._unwrapped = self._env.unwrapped

        self._num_envs = int(self._unwrapped.num_envs)
        self._device = str(self._unwrapped.device)
        self._base_obs_dim = self._infer_base_obs_dim()
        self._obs_dim = self._base_obs_dim + self.command_dim
        self._action_dim = self._infer_action_dim()
        self._imitation_obs_dim = self._infer_imitation_obs_dim()

        self._target_pose_bank = self._normalize_target_pose_bank(target_pose_bank)
        self._current_command: PasistCommand | None = None

    @property
    def obs_dim(self) -> int:
        """返回单个环境的 policy observation 维度。"""
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        """返回单个环境的动作维度。"""
        return self._action_dim

    @property
    def imitation_obs_dim(self) -> int:
        """返回 imitation observation 维度。"""
        return self._imitation_obs_dim

    @property
    def num_skills(self) -> int:
        """返回技能数量。"""
        return len(self._skill_names)

    @property
    def command_dim(self) -> int:
        """返回技能 command 的 one-hot 维度。"""
        return self._command_dim

    @property
    def velocity_range(self) -> tuple[float, float]:
        """覆盖基类默认速度采样范围。"""
        return self._velocity_range

    @property
    def skill_names(self) -> list[str]:
        """返回技能名称列表。"""
        return list(self._skill_names)

    @property
    def num_envs(self) -> int:
        """返回底层 Isaac Lab 并行环境数。"""
        return self._num_envs

    @property
    def task_id(self) -> str:
        """返回底层 Gym task id。"""
        return self._task_id

    def reset(
        self,
        command: PasistCommand | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        重置环境并返回 PASIST 友好的 observation / info。

        当前版本约定：
        - 如果不传 command，就按默认 skill 采样一个速度命令
        - 所有并行环境共享同一个 command
        """
        if command is None:
            command = self.sample_command(skill_id=self._default_skill_id)

        # Isaac Lab 的 reset 通常不强依赖 seed，这里优先尝试走 Gym 接口。
        try:
            reset_result = self._env.reset(seed=seed)
        except TypeError:
            reset_result = self._env.reset()

        raw_obs, raw_info = self._split_reset_result(reset_result)
        self._current_command = command
        self._apply_command_to_env(command)

        raw_policy_obs = self._extract_policy_observation(raw_obs)
        policy_obs = self._augment_policy_observation_with_command(raw_policy_obs, command)
        critic_obs = self._extract_critic_observation(raw_obs)
        measured_velocity = self._extract_measured_velocity(critic_obs)
        base_height = self._extract_base_height()
        base_pitch = self._extract_base_pitch()
        imitation_obs = self.extract_imitation_observation(policy_obs)

        self._ensure_target_pose_initialized(command.skill_id, policy_obs)
        info = self.build_info(
            observation=policy_obs,
            command=command,
            measured_velocity=measured_velocity,
            extra={
                "raw_obs": self._sanitize_to_host(raw_obs),
                "raw_info": self._sanitize_to_host(raw_info),
                "critic_obs": critic_obs.copy(),
                "imitation_obs": imitation_obs.copy(),
                "base_height": (
                    float(base_height) if np.asarray(base_height).ndim == 0 else np.asarray(base_height, dtype=np.float32).copy()
                ),
                "base_pitch": (
                    float(base_pitch) if np.asarray(base_pitch).ndim == 0 else np.asarray(base_pitch, dtype=np.float32).copy()
                ),
                "num_envs": self._num_envs,
                "task_id": self._task_id,
            },
        )

        return self._maybe_squeeze_obs(policy_obs), self._maybe_squeeze_info(info)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        执行动作并把 Isaac Lab 的 batched 结果翻译成 PASIST 友好的格式。

        当前实现支持：
        - 单环境输入 shape = [action_dim]
        - 多环境输入 shape = [num_envs, action_dim]
        """
        action_tensor = self._to_action_tensor(action)
        raw_obs, raw_reward, raw_terminated, raw_truncated, raw_info = self._env.step(action_tensor)

        # 为了尽量保持一个 episode 内 command 稳定，这里在 step 之后再次写回 command。
        # 这样即使底层 command term 的计时器到了，也会被我们的上层命令覆盖。
        if self._current_command is not None:
            self._apply_command_to_env(self._current_command)

        active_command = self._current_command or self.sample_command(skill_id=self._default_skill_id)
        raw_policy_obs = self._extract_policy_observation(raw_obs)
        policy_obs = self._augment_policy_observation_with_command(raw_policy_obs, active_command)
        critic_obs = self._extract_critic_observation(raw_obs)
        reward = self._tensor_to_numpy(raw_reward)
        terminated = self._tensor_to_numpy(raw_terminated).astype(bool)
        truncated = self._tensor_to_numpy(raw_truncated).astype(bool)
        measured_velocity = self._extract_measured_velocity(critic_obs)
        base_height = self._extract_base_height()
        base_pitch = self._extract_base_pitch()
        imitation_obs = self.extract_imitation_observation(policy_obs)
        info = self.build_info(
            observation=policy_obs,
            command=active_command,
            measured_velocity=measured_velocity,
            extra={
                "raw_obs": self._sanitize_to_host(raw_obs),
                "raw_info": self._sanitize_to_host(raw_info),
                "critic_obs": critic_obs.copy(),
                "imitation_obs": imitation_obs.copy(),
                "base_height": (
                    float(base_height) if np.asarray(base_height).ndim == 0 else np.asarray(base_height, dtype=np.float32).copy()
                ),
                "base_pitch": (
                    float(base_pitch) if np.asarray(base_pitch).ndim == 0 else np.asarray(base_pitch, dtype=np.float32).copy()
                ),
                "env_reward": reward.copy() if isinstance(reward, np.ndarray) else reward,
                "terminated": terminated.copy() if isinstance(terminated, np.ndarray) else terminated,
                "truncated": truncated.copy() if isinstance(truncated, np.ndarray) else truncated,
            },
        )

        return (
            self._maybe_squeeze_obs(policy_obs),
            self._maybe_squeeze_scalar(reward),
            self._maybe_squeeze_scalar(terminated),
            self._maybe_squeeze_scalar(truncated),
            self._maybe_squeeze_info(info),
        )

    def sample_command(self, skill_id: int | None = None) -> PasistCommand:
        """
        采样一个新的 PASIST command。

        当前实现先做最小版本：
        - 如果只配置了一个技能，skill_id 固定为 0
        - 如果以后配置多个技能，这里也已经支持按 skill_id 指定采样
        """
        if skill_id is None:
            skill_id = self._default_skill_id
        velocity = float(np.random.uniform(*self.velocity_range))
        return self.build_command(velocity=velocity, skill_id=int(skill_id))

    def get_target_pose(self, skill_id: int) -> np.ndarray:
        """
        返回某个技能对应的 target pose。

        当前版本要求每个 skill 对应一个定长向量。
        如果没有显式传入 target pose，则第一次 reset 时会自动初始化。
        """
        skill_id = int(skill_id)
        if skill_id not in self._target_pose_bank:
            raise KeyError(
                f"skill_id={skill_id} 还没有 target pose。"
                "请先 reset 一次自动初始化，或在构造函数中显式传入 target_pose_bank。"
            )
        return self._target_pose_bank[skill_id].copy()

    def extract_imitation_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        从完整 policy observation 中提取 imitation observation。

        当前返回 pose-only imitation observation：
        - base_height (1)
        - joint_pos_rel (12)
        - skill_id (1)

        这更贴近当前项目对 keyframe / target pose 的定义：
        target pose 不再包含速度；
        其中最后一位不再保存 pitch，而是保存离散 skill_id，
        方便后续模块直接通过 target pose 读取技能编号。
        """
        joint_pos_rel = self._extract_joint_pos_rel(observation)
        base_height = self._extract_base_height()
        current_skill_id = self._extract_current_skill_id()

        if np.asarray(joint_pos_rel).ndim == 1:
            imitation_obs = np.concatenate(
                [
                    np.asarray([base_height], dtype=np.float32).reshape(1),
                    np.asarray(joint_pos_rel, dtype=np.float32).reshape(-1),
                    np.asarray([current_skill_id], dtype=np.float32).reshape(1),
                ],
                axis=0,
            )
        else:
            height_column = np.asarray(base_height, dtype=np.float32).reshape(-1, 1)
            skill_id_column = np.asarray(current_skill_id, dtype=np.float32).reshape(-1, 1)
            imitation_obs = np.concatenate(
                [
                    height_column,
                    np.asarray(joint_pos_rel, dtype=np.float32),
                    skill_id_column,
                ],
                axis=-1,
            )
        return imitation_obs.astype(np.float32, copy=False)

    def measure_velocity(self, observation: np.ndarray) -> float:
        """
        从 observation 中估计速度。

        对当前 wrapper 来说，更可靠的速度来自 critic observation 的 base_lin_vel。
        这个方法主要保留基类接口兼容性，所以默认取：
        - batched 时第一个环境的 x 速度
        - 单环境时第一个分量
        """
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1 and obs.size > 0:
            return float(obs[0])
        if obs.ndim >= 2 and obs.shape[-1] > 0:
            return float(obs.reshape(-1, obs.shape[-1])[0, 0])
        return 0.0

    def close(self) -> None:
        """关闭底层 Gym / Isaac Lab 环境。"""
        self._env.close()

    def render(self, *args, **kwargs):
        """透传到底层环境的 render。"""
        return self._env.render(*args, **kwargs)

    def _register_builtin_tasks(self) -> None:
        """
        导入项目内已知环境模块，确保 Gym task 已完成注册。

        这里不做复杂逻辑，只做最基础的“导入即注册”。
        """
        importlib.import_module("envs.isaacsim_mini_envs")
        try:
            importlib.import_module("envs.pasist_env_cfg")
        except Exception:
            pass
        try:
            importlib.import_module("envs.go2")
        except Exception:
            # go2 任务不是 wrapper 能否工作的必要条件，失败时留给上层任务选择逻辑处理。
            pass

    def _resolve_env_cfg(self, task_id: str, env_cfg: Any | None, num_envs: int) -> Any:
        """
        解析用于 `gym.make` 的 env cfg。

        规则：
        - 如果调用方直接给了 env_cfg，就优先使用
        - 否则从 Gym 注册信息中的 env_cfg_entry_point 动态实例化
        """
        if env_cfg is not None:
            cfg = env_cfg
        else:
            spec = gym.spec(task_id)
            env_cfg_entry_point = spec.kwargs.get("env_cfg_entry_point")
            if env_cfg_entry_point is None:
                raise KeyError(f"{task_id} 没有 env_cfg_entry_point，无法自动创建 env cfg")
            module_name, class_name = env_cfg_entry_point.split(":")
            module = importlib.import_module(module_name)
            cfg = getattr(module, class_name)()

        if hasattr(cfg, "scene") and hasattr(cfg.scene, "num_envs"):
            cfg.scene.num_envs = int(num_envs)
        return cfg

    def _freeze_command_resampling(self, env_cfg: Any) -> None:
        """
        尽量关闭底层 velocity command 的随机重采样，让 command 更接近“由 PASIST 上层控制”。

        这是当前单技能版本里非常关键的一步，否则 Isaac Lab 自己会周期性重采样速度命令，
        使外层的 PasistCommand 和底层真实 command 漂移。
        """
        commands_cfg = getattr(env_cfg, "commands", None)
        if commands_cfg is None or not hasattr(commands_cfg, self._command_name):
            return

        command_cfg = getattr(commands_cfg, self._command_name)
        if hasattr(command_cfg, "resampling_time_range"):
            command_cfg.resampling_time_range = (1.0e9, 1.0e9)
        if hasattr(command_cfg, "rel_standing_envs"):
            command_cfg.rel_standing_envs = 0.0
        if hasattr(command_cfg, "debug_vis"):
            command_cfg.debug_vis = False

    def _infer_base_obs_dim(self) -> int:
        """从 observation_space 推断原始 policy observation 维度。"""
        obs_space = self._env.observation_space
        if hasattr(obs_space, "spaces"):
            policy_space = obs_space.spaces[self._policy_obs_key]
            return int(policy_space.shape[-1])
        return int(obs_space.shape[-1])

    def _infer_action_dim(self) -> int:
        """从 action_manager 或 action_space 推断单环境动作维度。"""
        action_dim = getattr(self._unwrapped.action_manager, "total_action_dim", None)
        if action_dim is not None:
            return int(action_dim)
        return int(self._env.action_space.shape[-1])

    def _infer_imitation_obs_dim(self) -> int:
        """当前 pose-only imitation observation 维度固定为 14。"""
        return 14

    def _normalize_skill_one_hot_map(
        self,
        skill_one_hot_map: dict[int, np.ndarray] | None,
    ) -> dict[int, np.ndarray]:
        """标准化 skill_id -> one-hot 映射。"""
        if skill_one_hot_map is None:
            return {}

        normalized: dict[int, np.ndarray] = {}
        dims: set[int] = set()
        for skill_id, one_hot in skill_one_hot_map.items():
            vector = np.asarray(one_hot, dtype=np.float32).reshape(-1)
            if vector.size == 0:
                raise ValueError(f"skill_id={skill_id} 的 one-hot 不能为空")
            normalized[int(skill_id)] = vector.copy()
            dims.add(int(vector.size))
        if len(dims) > 1:
            raise ValueError("skill_one_hot_map 中所有 one-hot 维度必须一致")
        return normalized

    def _infer_command_dim(self) -> int:
        """推断技能 command 维度。"""
        if self._skill_one_hot_map:
            first_vector = next(iter(self._skill_one_hot_map.values()))
            return int(first_vector.shape[0])
        return int(self.num_skills)

    def _validate_skill_one_hot_coverage(self) -> None:
        """确认当前启用的技能在 one-hot 配置中都有定义。"""
        if not self._skill_one_hot_map:
            return
        for skill_id in range(self.num_skills):
            if skill_id not in self._skill_one_hot_map:
                raise KeyError(
                    f"skill_one_hot_map 中缺少 skill_id={skill_id} 的编码；"
                    f"当前启用的技能为 {self._skill_names}"
                )

    def get_skill_one_hot(self, skill_id: int) -> np.ndarray:
        """返回指定技能对应的 one-hot 编码。"""
        skill_id = int(skill_id)
        if self._skill_one_hot_map:
            if skill_id not in self._skill_one_hot_map:
                raise KeyError(f"skill_id={skill_id} 没有在 skill_one_hot_map 中定义")
            return self._skill_one_hot_map[skill_id].copy()
        return super().get_skill_one_hot(skill_id)

    def _normalize_target_pose_bank(
        self,
        target_pose_bank: Sequence[np.ndarray] | dict[int, np.ndarray] | None,
    ) -> dict[int, np.ndarray]:
        """
        标准化 target pose bank。

        支持两种输入形式：
        - `dict[int, np.ndarray]`
        - `list[np.ndarray]` / `tuple[np.ndarray]`
        """
        if target_pose_bank is None:
            return {}

        normalized: dict[int, np.ndarray] = {}
        if isinstance(target_pose_bank, dict):
            items = target_pose_bank.items()
        else:
            items = enumerate(target_pose_bank)

        for skill_id, target_pose in items:
            pose = np.asarray(target_pose, dtype=np.float32).reshape(-1)
            if pose.shape[0] != self._imitation_obs_dim:
                raise ValueError(
                    f"target_pose_bank 中 skill_id={skill_id} 的维度不正确："
                    f"期望 {self._imitation_obs_dim}，实际得到 {pose.shape[0]}。"
                    "当前工程的 target pose 约定为 [base_height, joint_pos_rel(12), skill_id] 共 14 维。"
                )
            encoded_skill_id = int(round(float(pose[-1])))
            if encoded_skill_id != int(skill_id):
                raise ValueError(
                    f"target_pose_bank 中 skill_id={skill_id} 的最后一维编码为 {encoded_skill_id}，两者不一致。"
                )
            normalized[int(skill_id)] = pose.copy()
        return normalized

    def _ensure_target_pose_initialized(self, skill_id: int, policy_obs: np.ndarray) -> None:
        """
        如果某个 skill 还没有 target pose，就用当前观测自动初始化。

        这对最小单技能版本特别实用：
        - 不需要你一开始就准备 motion reference
        - 可以先用机器人初始姿态当作一个稳定的 target pose
        """
        skill_id = int(skill_id)
        if skill_id in self._target_pose_bank:
            return

        imitation_obs = self.extract_imitation_observation(policy_obs)
        if imitation_obs.ndim > 1:
            pose = imitation_obs[0]
        else:
            pose = imitation_obs
        self._target_pose_bank[skill_id] = np.asarray(pose, dtype=np.float32).reshape(-1).copy()

    def _extract_policy_observation(self, raw_obs: Any) -> np.ndarray:
        """从 Isaac Lab 的 observation 返回值中提取原始 policy observation。"""
        if isinstance(raw_obs, dict):
            policy_obs = raw_obs[self._policy_obs_key]
        else:
            policy_obs = raw_obs
        return self._tensor_to_numpy(policy_obs).astype(np.float32, copy=False)

    def _augment_policy_observation_with_command(
        self,
        policy_obs: np.ndarray,
        command: PasistCommand,
    ) -> np.ndarray:
        """
        在原始 policy observation 末尾拼接技能 one-hot command。

        这样 actor 输入会从原来的 45 维扩展为 49 维，
        且 skill command 的真来源统一为环境侧的 `PasistCommand.one_hot`。
        """
        obs = np.asarray(policy_obs, dtype=np.float32)
        command_one_hot = np.asarray(command.one_hot, dtype=np.float32).reshape(-1)
        if command_one_hot.shape[0] != self.command_dim:
            raise ValueError(
                f"command one-hot 维度不正确：期望 {self.command_dim}，实际得到 {command_one_hot.shape[0]}"
            )

        if obs.ndim == 1:
            return np.concatenate([obs, command_one_hot], axis=0).astype(np.float32, copy=False)

        command_batch = np.broadcast_to(command_one_hot.reshape(1, -1), (obs.shape[0], command_one_hot.shape[0]))
        return np.concatenate([obs, command_batch.astype(np.float32, copy=False)], axis=-1).astype(np.float32, copy=False)

    def _extract_critic_observation(self, raw_obs: Any) -> np.ndarray:
        """从 Isaac Lab observation 中提取 critic observation；如果没有则回退到 policy。"""
        if isinstance(raw_obs, dict) and self._critic_obs_key in raw_obs:
            critic_obs = raw_obs[self._critic_obs_key]
            return self._tensor_to_numpy(critic_obs).astype(np.float32, copy=False)
        return self._extract_policy_observation(raw_obs)

    def _extract_measured_velocity(self, critic_obs: np.ndarray) -> float | np.ndarray:
        """
        从 critic observation 中提取测得速度。

        对当前最小环境来说，critic observation 的前 3 维就是 `base_lin_vel`，
        这里优先取其中的 x 方向速度，供 task reward / info 使用。
        """
        obs = np.asarray(critic_obs, dtype=np.float32)
        if obs.ndim == 1:
            return float(obs[0]) if obs.size > 0 else 0.0
        if obs.ndim == 2:
            return obs[:, 0].astype(np.float32, copy=False)
        flattened = obs.reshape(-1, obs.shape[-1])
        return flattened[:, 0].astype(np.float32, copy=False)

    def _extract_joint_pos_rel(self, policy_obs: np.ndarray) -> np.ndarray:
        """
        从 policy observation 中提取 joint_pos_rel。

        当前 Go2 最小环境的 policy observation 结构中：
        - [9:21] -> joint_pos_rel
        """
        obs = np.asarray(policy_obs, dtype=np.float32)
        start, end = self._joint_pos_rel_slice
        if obs.shape[-1] < end:
            raise ValueError(f"policy observation 维度不足，无法提取 joint_pos_rel；需要至少 {end} 维")
        return obs[..., start:end].astype(np.float32, copy=False)

    def _extract_base_height(self) -> float | np.ndarray:
        """
        直接从 IsaacLab 机器人状态读取 base height。

        这里读取的是 `robot.data.root_pos_w[:, 2]`，也就是世界坐标系下的根部高度。
        """
        robot = self._unwrapped.scene["robot"]
        base_height = self._tensor_to_numpy(robot.data.root_pos_w[:, 2]).astype(np.float32, copy=False)
        if self._num_envs == 1 and base_height.shape[0] == 1:
            return float(base_height[0])
        return base_height

    def _extract_base_pitch(self) -> float | np.ndarray:
        """
        直接从 IsaacLab 机器人状态读取 base pitch。

        这里通过根部四元数 `root_quat_w` 转换到 Euler XYZ，取中间的 pitch 分量。
        """
        import isaaclab.utils.math as math_utils

        robot = self._unwrapped.scene["robot"]
        root_quat_w = robot.data.root_quat_w
        _, pitch, _ = math_utils.euler_xyz_from_quat(root_quat_w)
        pitch_array = self._tensor_to_numpy(pitch).astype(np.float32, copy=False)
        if self._num_envs == 1 and pitch_array.shape[0] == 1:
            return float(pitch_array[0])
        return pitch_array

    def _extract_current_skill_id(self) -> float | np.ndarray:
        """
        返回当前 command 对应的 skill_id，并转成可拼接到 imitation observation 的浮点表示。

        当前训练流程里所有并行环境共享同一个 skill command，
        因此这里会广播成 `[num_envs]`。
        """
        active_command = self._current_command
        skill_id = float(self._default_skill_id if active_command is None else active_command.skill_id)
        if self._num_envs == 1:
            return skill_id
        return np.full((self._num_envs,), skill_id, dtype=np.float32)

    def _split_reset_result(self, reset_result: Any) -> tuple[Any, dict[str, Any]]:
        """兼容 `env.reset()` 可能返回 `obs` 或 `(obs, info)` 两种形式。"""
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            return reset_result[0], reset_result[1]
        return reset_result, {}

    def _apply_command_to_env(self, command: PasistCommand) -> None:
        """
        把上层 PasistCommand 写入底层 Isaac Lab command term。

        当前只覆盖 velocity command：
        - x 速度由 command.velocity 提供
        - y 速度固定 0
        - yaw 速度固定 0

        后面扩展多技能时，可以按 skill_id 决定不同的 command 映射策略。
        """
        try:
            command_term = self._unwrapped.command_manager.get_term(self._command_name)
        except Exception:
            return

        # UniformVelocityCommand 内部实际把命令保存在 vel_command_b 中。
        # 这里直接覆写这块 tensor，是最小版本里最直接、最稳定的办法。
        vel_command = getattr(command_term, "vel_command_b", None)
        if vel_command is None:
            return

        vel_command[:, :] = 0.0
        vel_command[:, 0] = float(command.velocity)

        if hasattr(command_term, "is_standing_env"):
            command_term.is_standing_env[:] = False
        if hasattr(command_term, "is_heading_env"):
            command_term.is_heading_env[:] = False

    def _to_action_tensor(self, action: np.ndarray):
        """
        把 numpy action 转成 Isaac Lab 期望的 torch.Tensor。

        支持：
        - 单环境：shape = [action_dim]
        - 多环境：shape = [num_envs, action_dim]
        """
        import torch

        action_array = np.asarray(action, dtype=np.float32)
        if action_array.ndim == 1:
            if self._num_envs != 1:
                action_array = np.broadcast_to(action_array[None, :], (self._num_envs, self._action_dim))
            else:
                action_array = action_array.reshape(1, self._action_dim)
        elif action_array.ndim != 2:
            raise ValueError(f"action 维度不合法，期望 1D 或 2D，实际为 {action_array.ndim}D")

        if action_array.shape != (self._num_envs, self._action_dim):
            raise ValueError(
                f"action shape 不匹配，期望 {(self._num_envs, self._action_dim)}，实际为 {action_array.shape}"
            )

        return torch.as_tensor(action_array, dtype=torch.float32, device=self._unwrapped.device)

    def _tensor_to_numpy(self, value: Any) -> np.ndarray:
        """把 torch tensor 或 numpy / 标量统一转成 numpy。"""
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return np.asarray(value)

    def _sanitize_to_host(self, value: Any) -> Any:
        """
        递归地把 Isaac Lab / PyTorch 返回的对象整理成主机端可安全访问的结构。

        作用:
        - 避免 `info["raw_info"]` 之类的调试字段里残留 CUDA tensor
        - 后续即使上层代码或日志逻辑访问这些字段，也不会触发
          `can't convert cuda tensor to numpy` 这一类错误
        """
        if isinstance(value, dict):
            return {key: self._sanitize_to_host(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self._sanitize_to_host(item) for item in value)
        if isinstance(value, list):
            return [self._sanitize_to_host(item) for item in value]
        if hasattr(value, "detach") or hasattr(value, "cpu"):
            return self._tensor_to_numpy(value)
        return value

    def _maybe_squeeze_obs(self, observation: np.ndarray) -> np.ndarray:
        """当 `num_envs == 1` 时，把 batched observation 挤压成一维。"""
        if self._num_envs == 1 and observation.ndim >= 2 and observation.shape[0] == 1:
            return observation[0]
        return observation

    def _maybe_squeeze_scalar(self, value: Any) -> Any:
        """当 `num_envs == 1` 时，把 batched reward / done 挤压成标量。"""
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == 1:
            scalar = value[0]
            if isinstance(scalar, np.generic):
                return scalar.item()
            return scalar
        return value

    def _maybe_squeeze_info(self, info: dict[str, Any]) -> dict[str, Any]:
        """
        对 info 中显然是按环境批处理的字段做单环境挤压。

        这里只处理最常见的 numpy 数组字段，保留原始 `raw_obs/raw_info`
        方便调试底层 Isaac Lab 行为。
        """
        if self._num_envs != 1:
            return info

        squeezed: dict[str, Any] = {}
        for key, value in info.items():
            if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == 1:
                squeezed[key] = value[0]
            else:
                squeezed[key] = value
        return squeezed
