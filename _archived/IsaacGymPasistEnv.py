from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

from envs.BasePasistEnv import BasePasistEnv, PasistCommand


class IsaacGymPasistEnv(BasePasistEnv):
    """
    Isaac Gym / IsaacGymEnvs 后端到当前 PASIST 训练接口的适配器。

    这个类刻意不直接绑定某一个 Isaac Gym task 实现，而是包装一个
    VecTask 风格的后端对象。后端通常需要提供：
    - reset() / step(actions)
    - num_envs
    - num_actions 或 action_space
    - obs_buf / states_buf，或 reset/step 返回对应 observation
    - root_states / dof_pos / default_dof_pos 等状态张量

    这样做的目标是先保住当前工程已经稳定的 PPO / SIL / reward 代码，
    后续你可以把真正的 Go2 Isaac Gym task 接到 backend_factory 里。
    """

    def __init__(
        self,
        backend: Any | None = None,
        backend_factory: Callable[..., Any] | str | None = None,
        backend_kwargs: Mapping[str, Any] | None = None,
        num_envs: int | None = None,
        skill_names: Sequence[str] | None = None,
        target_pose_bank: Sequence[np.ndarray] | dict[int, np.ndarray] | None = None,
        skill_one_hot_map: dict[int, np.ndarray] | None = None,
        default_skill_id: int = 0,
        velocity_range: tuple[float, float] = (-0.5, 0.5),
        policy_obs_key: str = "obs",
        critic_obs_key: str = "states",
        joint_pos_rel_slice: tuple[int, int] = (9, 21),
        command_attr_candidates: Sequence[str] = ("commands", "commands_buf", "command_buf"),
        root_state_attr_candidates: Sequence[str] = ("root_states", "root_state_tensor", "root_states_tensor"),
        dof_pos_attr_candidates: Sequence[str] = ("dof_pos", "dof_pos_tensor"),
        default_dof_pos_attr_candidates: Sequence[str] = (
            "default_dof_pos",
            "default_dof_pos_tensor",
            "default_joint_pos",
        ),
        root_quat_order: str = "xyzw",
    ) -> None:
        """
        参数说明：
        - backend:
          已经构造好的 Isaac Gym VecTask 风格对象。
        - backend_factory:
          可调用对象，或形如 "package.module:function" 的字符串。未提供 backend
          时会调用它来构造后端。
        - backend_kwargs:
          传给 backend_factory 的参数。
        - num_envs:
          可选覆盖并行环境数量。若后端支持 cfg 字段，建议在 factory 内处理。
        - policy_obs_key / critic_obs_key:
          IsaacGymEnvs 常见返回键为 "obs" / "states"。如果你的 task 返回
          "policy" / "critic"，这里也可以直接改。
        - root_quat_order:
          Isaac Gym root state 通常是 xyzw；如果你的后端已经转成 wxyz，可传 "wxyz"。
        """
        self._backend = backend or self._build_backend(backend_factory, backend_kwargs)
        if self._backend is None:
            raise ValueError(
                "IsaacGymPasistEnv 需要 backend 或 backend_factory。"
                "先传入一个 IsaacGymEnvs/VecTask 风格后端，再由本适配器接入 PASIST trainer。"
            )

        if skill_names is None:
            skill_names = ["walk"]
        self._skill_names = [str(name) for name in skill_names]
        if not self._skill_names:
            raise ValueError("skill_names 不能为空")

        self._default_skill_id = int(default_skill_id)
        if not 0 <= self._default_skill_id < len(self._skill_names):
            raise ValueError(f"default_skill_id 超出范围: {self._default_skill_id}")

        self._velocity_range = (float(velocity_range[0]), float(velocity_range[1]))
        self._policy_obs_key = str(policy_obs_key)
        self._critic_obs_key = str(critic_obs_key)
        self._joint_pos_rel_slice = (int(joint_pos_rel_slice[0]), int(joint_pos_rel_slice[1]))
        self._command_attr_candidates = tuple(str(item) for item in command_attr_candidates)
        self._root_state_attr_candidates = tuple(str(item) for item in root_state_attr_candidates)
        self._dof_pos_attr_candidates = tuple(str(item) for item in dof_pos_attr_candidates)
        self._default_dof_pos_attr_candidates = tuple(str(item) for item in default_dof_pos_attr_candidates)
        self._root_quat_order = str(root_quat_order).lower()
        if self._root_quat_order not in {"xyzw", "wxyz"}:
            raise ValueError("root_quat_order 只支持 'xyzw' 或 'wxyz'")

        self._num_envs = self._infer_num_envs(num_envs)
        self._device = self._infer_backend_device()
        self._base_obs_dim = self._infer_base_obs_dim()
        self._base_critic_obs_dim = self._infer_base_critic_obs_dim()
        self._action_dim = self._infer_action_dim()
        self._imitation_obs_dim = 14

        self._skill_one_hot_map = self._normalize_skill_one_hot_map(skill_one_hot_map)
        self._command_dim = self._infer_command_dim()
        self._validate_skill_one_hot_coverage()
        self._obs_dim = self._base_obs_dim + self._command_dim
        self._critic_obs_dim = self._base_critic_obs_dim + self._command_dim

        self._target_pose_bank = self._normalize_target_pose_bank(target_pose_bank)
        self._current_command: PasistCommand | None = None
        self._last_raw_obs: Any | None = None

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def critic_obs_dim(self) -> int:
        return self._critic_obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def imitation_obs_dim(self) -> int:
        return self._imitation_obs_dim

    @property
    def num_skills(self) -> int:
        return len(self._skill_names)

    @property
    def command_dim(self) -> int:
        return self._command_dim

    @property
    def velocity_range(self) -> tuple[float, float]:
        return self._velocity_range

    @property
    def skill_names(self) -> list[str]:
        return list(self._skill_names)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> str:
        return self._device

    @property
    def backend(self) -> Any:
        return self._backend

    def reset(
        self,
        command: PasistCommand | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._seed_backend(seed)

        if command is None:
            command = self.sample_command(skill_id=self._default_skill_id)
        self._current_command = command
        self._apply_command_to_backend(command)

        raw_reset = self._backend.reset()
        raw_obs, raw_info = self._split_reset_result(raw_reset)
        self._last_raw_obs = raw_obs

        raw_policy_obs = self._extract_policy_observation(raw_obs)
        policy_obs = self._augment_policy_observation_with_command(raw_policy_obs, command)
        self._ensure_target_pose_initialized(command.skill_id, policy_obs)

        info = self._build_step_info(
            policy_obs=policy_obs,
            raw_obs=raw_obs,
            raw_info=raw_info,
            command=command,
        )
        return self._maybe_squeeze_obs(policy_obs), self._maybe_squeeze_info(info)

    def reset_tensor(
        self,
        command: PasistCommand | None = None,
        seed: int | None = None,
        lightweight_info: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if seed is not None:
            self._seed_backend(seed)

        if command is None:
            command = self.sample_command(skill_id=self._default_skill_id)
        self._current_command = command
        self._apply_command_to_backend(command)

        raw_reset = self._backend.reset()
        raw_obs, raw_info = self._split_reset_result(raw_reset)
        self._last_raw_obs = raw_obs

        raw_policy_obs = self._extract_policy_observation_tensor(raw_obs)
        policy_obs = self._augment_policy_observation_with_command_tensor(raw_policy_obs, command)
        self._ensure_target_pose_initialized(command.skill_id, self._tensor_to_numpy(policy_obs))

        info = self._build_step_info_tensor(
            policy_obs=policy_obs,
            raw_obs=raw_obs,
            raw_info=raw_info,
            command=command,
            lightweight=lightweight_info,
        )
        return self._maybe_squeeze_tensor_obs(policy_obs), self._maybe_squeeze_tensor_info(info)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        active_command = self._current_command or self.sample_command(skill_id=self._default_skill_id)
        self._apply_command_to_backend(active_command)

        raw_step = self._backend.step(self._to_action_tensor(action))
        raw_obs, raw_reward, raw_done, raw_info, raw_truncated = self._split_step_result(raw_step)
        self._last_raw_obs = raw_obs

        # 有些 IsaacGymEnvs task 会在 step 内更新 command，这里再次写回上层 command。
        self._apply_command_to_backend(active_command)

        raw_policy_obs = self._extract_policy_observation(raw_obs)
        policy_obs = self._augment_policy_observation_with_command(raw_policy_obs, active_command)
        reward = self._tensor_to_numpy(raw_reward).astype(np.float32, copy=False)
        done = self._tensor_to_numpy(raw_done).astype(bool)

        if raw_truncated is None:
            truncated = self._extract_timeout_flags(raw_info, done)
        else:
            truncated = self._tensor_to_numpy(raw_truncated).astype(bool)
        terminated = np.logical_and(done, np.logical_not(truncated))

        info = self._build_step_info(
            policy_obs=policy_obs,
            raw_obs=raw_obs,
            raw_info=raw_info,
            command=active_command,
            env_reward=reward,
            terminated=terminated,
            truncated=truncated,
        )

        return (
            self._maybe_squeeze_obs(policy_obs),
            self._maybe_squeeze_scalar(reward),
            self._maybe_squeeze_scalar(terminated),
            self._maybe_squeeze_scalar(truncated),
            self._maybe_squeeze_info(info),
        )

    def step_tensor(
        self,
        action: torch.Tensor | np.ndarray,
        lightweight_info: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        active_command = self._current_command or self.sample_command(skill_id=self._default_skill_id)
        self._apply_command_to_backend(active_command)

        raw_step = self._backend.step(self._to_action_tensor(action))
        raw_obs, raw_reward, raw_done, raw_info, raw_truncated = self._split_step_result_tensor(raw_step)
        self._last_raw_obs = raw_obs

        self._apply_command_to_backend(active_command)

        raw_policy_obs = self._extract_policy_observation_tensor(raw_obs)
        policy_obs = self._augment_policy_observation_with_command_tensor(raw_policy_obs, active_command)
        reward = self._to_backend_float_tensor(raw_reward)
        done = self._to_backend_bool_tensor(raw_done)

        if raw_truncated is None:
            truncated = self._extract_timeout_flags_tensor(raw_info, done)
        else:
            truncated = self._to_backend_bool_tensor(raw_truncated)
        terminated = torch.logical_and(done, torch.logical_not(truncated))

        info = self._build_step_info_tensor(
            policy_obs=policy_obs,
            raw_obs=raw_obs,
            raw_info=raw_info,
            command=active_command,
            env_reward=reward,
            terminated=terminated,
            truncated=truncated,
            lightweight=lightweight_info,
        )

        return (
            self._maybe_squeeze_tensor_obs(policy_obs),
            self._maybe_squeeze_tensor_scalar(reward),
            self._maybe_squeeze_tensor_scalar(terminated),
            self._maybe_squeeze_tensor_scalar(truncated),
            self._maybe_squeeze_tensor_info(info),
        )

    def sample_command(self, skill_id: int | None = None) -> PasistCommand:
        if skill_id is None:
            skill_id = self._default_skill_id
        velocity = float(np.random.uniform(*self.velocity_range))
        return self.build_command(velocity=velocity, skill_id=int(skill_id))

    def get_target_pose(self, skill_id: int) -> np.ndarray:
        skill_id = int(skill_id)
        if skill_id not in self._target_pose_bank:
            raise KeyError(
                f"skill_id={skill_id} 还没有 target pose。"
                "请先 reset 一次自动初始化，或在构造函数中显式传入 target_pose_bank。"
            )
        return self._target_pose_bank[skill_id].copy()

    def get_target_pose_tensor(self, skill_id: int) -> torch.Tensor:
        return self._to_backend_float_tensor(self.get_target_pose(skill_id))

    def extract_imitation_observation(self, observation: np.ndarray) -> np.ndarray:
        joint_pos_rel = self._extract_joint_pos_rel(observation)
        base_height = self._extract_base_height()
        current_skill_id = self._extract_current_skill_id()

        if np.asarray(joint_pos_rel).ndim == 1:
            return np.concatenate(
                [
                    np.asarray([base_height], dtype=np.float32).reshape(1),
                    np.asarray(joint_pos_rel, dtype=np.float32).reshape(-1),
                    np.asarray([current_skill_id], dtype=np.float32).reshape(1),
                ],
                axis=0,
            ).astype(np.float32, copy=False)

        height_column = np.asarray(base_height, dtype=np.float32).reshape(-1, 1)
        skill_id_column = np.asarray(current_skill_id, dtype=np.float32).reshape(-1, 1)
        return np.concatenate(
            [
                height_column,
                np.asarray(joint_pos_rel, dtype=np.float32),
                skill_id_column,
            ],
            axis=-1,
        ).astype(np.float32, copy=False)

    def extract_imitation_observation_tensor(self, observation: torch.Tensor) -> torch.Tensor:
        joint_pos_rel = self._extract_joint_pos_rel_tensor(observation)
        base_height = self._extract_base_height_tensor()
        current_skill_id = self._extract_current_skill_id_tensor()

        if joint_pos_rel.ndim == 1:
            return torch.cat(
                [
                    base_height.reshape(1).to(dtype=torch.float32),
                    joint_pos_rel.reshape(-1).to(dtype=torch.float32),
                    current_skill_id.reshape(1).to(dtype=torch.float32),
                ],
                dim=0,
            )

        height_column = base_height.reshape(-1, 1).to(dtype=torch.float32)
        skill_id_column = current_skill_id.reshape(-1, 1).to(dtype=torch.float32)
        return torch.cat(
            [
                height_column,
                joint_pos_rel.to(dtype=torch.float32),
                skill_id_column,
            ],
            dim=-1,
        )

    def close(self) -> None:
        close_fn = getattr(self._backend, "close", None)
        if callable(close_fn):
            close_fn()

    def render(self, *args, **kwargs):
        render_fn = getattr(self._backend, "render", None)
        if callable(render_fn):
            return render_fn(*args, **kwargs)
        return None

    def get_skill_one_hot(self, skill_id: int) -> np.ndarray:
        skill_id = int(skill_id)
        if self._skill_one_hot_map:
            if skill_id not in self._skill_one_hot_map:
                raise KeyError(f"skill_id={skill_id} 没有在 skill_one_hot_map 中定义")
            return self._skill_one_hot_map[skill_id].copy()
        return super().get_skill_one_hot(skill_id)

    def _build_backend(
        self,
        backend_factory: Callable[..., Any] | str | None,
        backend_kwargs: Mapping[str, Any] | None,
    ) -> Any | None:
        if backend_factory is None:
            return None
        kwargs = dict(backend_kwargs or {})
        if isinstance(backend_factory, str):
            module_name, _, attr_name = backend_factory.partition(":")
            if not module_name or not attr_name:
                raise ValueError("backend_factory 字符串必须形如 'package.module:function'")
            module = importlib.import_module(module_name)
            backend_factory = getattr(module, attr_name)
        if not callable(backend_factory):
            raise TypeError("backend_factory 必须是可调用对象或 'module:function' 字符串")
        return backend_factory(**kwargs)

    def _seed_backend(self, seed: int) -> None:
        for method_name in ("seed", "set_seed"):
            method = getattr(self._backend, method_name, None)
            if callable(method):
                method(int(seed))
                return

    def _infer_num_envs(self, override: int | None) -> int:
        if override is not None:
            return int(override)
        for attr_name in ("num_envs", "numEnvs"):
            value = getattr(self._backend, attr_name, None)
            if value is not None:
                return int(value)
        obs = self._peek_backend_observation()
        if obs is not None:
            return int(np.asarray(obs).reshape(np.asarray(obs).shape[0], -1).shape[0])
        return 1

    def _infer_backend_device(self) -> str:
        for attr_name in ("rl_device", "device", "sim_device"):
            value = getattr(self._backend, attr_name, None)
            if value is not None:
                return str(value)
        return "cpu"

    def _infer_base_obs_dim(self) -> int:
        for attr_name in ("num_obs", "numObservations"):
            value = getattr(self._backend, attr_name, None)
            if value is not None:
                return int(value)

        observation_space = getattr(self._backend, "observation_space", None)
        shape = getattr(observation_space, "shape", None)
        if shape:
            return int(shape[-1])

        obs = self._peek_backend_observation()
        if obs is None:
            raise AttributeError("无法从 backend 推断 observation 维度")
        return int(np.asarray(obs).shape[-1])

    def _infer_base_critic_obs_dim(self) -> int:
        for attr_name in ("num_states", "numStates"):
            value = getattr(self._backend, attr_name, None)
            if value is not None:
                return int(value)

        critic_obs = self._peek_backend_critic_observation()
        if critic_obs is not None:
            return int(np.asarray(critic_obs).shape[-1])

        return self._base_obs_dim

    def _infer_action_dim(self) -> int:
        for attr_name in ("num_actions", "numActions"):
            value = getattr(self._backend, attr_name, None)
            if value is not None:
                return int(value)

        action_space = getattr(self._backend, "action_space", None)
        shape = getattr(action_space, "shape", None)
        if shape:
            return int(shape[-1])

        raise AttributeError("无法从 backend 推断 action 维度；请确保后端提供 num_actions 或 action_space")

    def _peek_backend_observation(self) -> np.ndarray | None:
        for attr_name in ("obs_buf", "obs", "observations"):
            value = getattr(self._backend, attr_name, None)
            if value is not None:
                return self._tensor_to_numpy(value).astype(np.float32, copy=False)
        return None

    def _peek_backend_critic_observation(self) -> np.ndarray | None:
        for attr_name in ("states_buf", "states", "critic_observations"):
            value = getattr(self._backend, attr_name, None)
            if value is not None:
                return self._tensor_to_numpy(value).astype(np.float32, copy=False)
        return None

    def _normalize_skill_one_hot_map(
        self,
        skill_one_hot_map: dict[int, np.ndarray] | None,
    ) -> dict[int, np.ndarray]:
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
        if self._skill_one_hot_map:
            first_vector = next(iter(self._skill_one_hot_map.values()))
            return int(first_vector.shape[0])
        return int(self.num_skills)

    def _validate_skill_one_hot_coverage(self) -> None:
        if not self._skill_one_hot_map:
            return
        for skill_id in range(self.num_skills):
            if skill_id not in self._skill_one_hot_map:
                raise KeyError(
                    f"skill_one_hot_map 中缺少 skill_id={skill_id} 的编码；"
                    f"当前启用的技能为 {self._skill_names}"
                )

    def _normalize_target_pose_bank(
        self,
        target_pose_bank: Sequence[np.ndarray] | dict[int, np.ndarray] | None,
    ) -> dict[int, np.ndarray]:
        if target_pose_bank is None:
            return {}

        normalized: dict[int, np.ndarray] = {}
        items = target_pose_bank.items() if isinstance(target_pose_bank, dict) else enumerate(target_pose_bank)
        for skill_id, target_pose in items:
            pose = np.asarray(target_pose, dtype=np.float32).reshape(-1)
            if pose.shape[0] != self._imitation_obs_dim:
                raise ValueError(
                    f"target_pose_bank 中 skill_id={skill_id} 的维度不正确："
                    f"期望 {self._imitation_obs_dim}，实际得到 {pose.shape[0]}。"
                    "当前 target pose 约定为 [base_height, joint_pos_rel(12), skill_id] 共 14 维。"
                )
            encoded_skill_id = int(round(float(pose[-1])))
            if encoded_skill_id != int(skill_id):
                raise ValueError(
                    f"target_pose_bank 中 skill_id={skill_id} 的最后一维编码为 {encoded_skill_id}，两者不一致。"
                )
            normalized[int(skill_id)] = pose.copy()
        return normalized

    def _ensure_target_pose_initialized(self, skill_id: int, policy_obs: np.ndarray) -> None:
        skill_id = int(skill_id)
        if skill_id in self._target_pose_bank:
            return
        imitation_obs = self.extract_imitation_observation(policy_obs)
        pose = imitation_obs[0] if imitation_obs.ndim > 1 else imitation_obs
        self._target_pose_bank[skill_id] = np.asarray(pose, dtype=np.float32).reshape(-1).copy()

    def _split_reset_result(self, reset_result: Any) -> tuple[Any, dict[str, Any]]:
        if isinstance(reset_result, tuple):
            if len(reset_result) == 2 and isinstance(reset_result[1], Mapping):
                return reset_result[0], dict(reset_result[1])
            if len(reset_result) >= 1:
                return reset_result[0], {}
        return reset_result, {}

    def _split_step_result(self, step_result: Any) -> tuple[Any, Any, Any, dict[str, Any], Any | None]:
        if not isinstance(step_result, tuple):
            raise TypeError("backend.step(action) 必须返回 tuple")

        if len(step_result) == 4:
            raw_obs, raw_reward, raw_done, raw_info = step_result
            return raw_obs, raw_reward, raw_done, self._normalize_info(raw_info), None
        if len(step_result) == 5:
            raw_obs, raw_reward, raw_terminated, raw_truncated, raw_info = step_result
            info = self._normalize_info(raw_info)
            done = np.logical_or(
                self._tensor_to_numpy(raw_terminated).astype(bool),
                self._tensor_to_numpy(raw_truncated).astype(bool),
            )
            return raw_obs, raw_reward, done, info, raw_truncated

        raise ValueError(f"backend.step(action) 返回长度不支持：{len(step_result)}")

    def _split_step_result_tensor(self, step_result: Any) -> tuple[Any, Any, Any, dict[str, Any], Any | None]:
        if not isinstance(step_result, tuple):
            raise TypeError("backend.step(action) 必须返回 tuple")

        if len(step_result) == 4:
            raw_obs, raw_reward, raw_done, raw_info = step_result
            return raw_obs, raw_reward, raw_done, self._normalize_info(raw_info), None
        if len(step_result) == 5:
            raw_obs, raw_reward, raw_terminated, raw_truncated, raw_info = step_result
            info = self._normalize_info(raw_info)
            done = torch.logical_or(
                self._to_backend_bool_tensor(raw_terminated),
                self._to_backend_bool_tensor(raw_truncated),
            )
            return raw_obs, raw_reward, done, info, raw_truncated

        raise ValueError(f"backend.step(action) 返回长度不支持：{len(step_result)}")

    def _normalize_info(self, info: Any) -> dict[str, Any]:
        if info is None:
            return {}
        if isinstance(info, Mapping):
            return dict(info)
        return {"raw_extras": info}

    def _extract_timeout_flags(self, raw_info: dict[str, Any], done: np.ndarray) -> np.ndarray:
        for key in ("time_outs", "timeouts", "timeout", "truncated"):
            if key in raw_info:
                return self._tensor_to_numpy(raw_info[key]).astype(bool)
        return np.zeros_like(done, dtype=bool)

    def _extract_timeout_flags_tensor(self, raw_info: dict[str, Any], done: torch.Tensor) -> torch.Tensor:
        for key in ("time_outs", "timeouts", "timeout", "truncated"):
            if key in raw_info:
                return self._to_backend_bool_tensor(raw_info[key])
        return torch.zeros_like(done, dtype=torch.bool)

    def _extract_policy_observation(self, raw_obs: Any) -> np.ndarray:
        if isinstance(raw_obs, Mapping):
            for key in (self._policy_obs_key, "policy", "obs"):
                if key in raw_obs:
                    return self._tensor_to_numpy(raw_obs[key]).astype(np.float32, copy=False)
        return self._tensor_to_numpy(raw_obs).astype(np.float32, copy=False)

    def _extract_policy_observation_tensor(self, raw_obs: Any) -> torch.Tensor:
        if isinstance(raw_obs, Mapping):
            for key in (self._policy_obs_key, "policy", "obs"):
                if key in raw_obs:
                    return self._to_backend_float_tensor(raw_obs[key])
        return self._to_backend_float_tensor(raw_obs)

    def _extract_critic_observation(self, raw_obs: Any) -> np.ndarray:
        if isinstance(raw_obs, Mapping):
            for key in (self._critic_obs_key, "critic", "states", "obs"):
                if key in raw_obs:
                    return self._tensor_to_numpy(raw_obs[key]).astype(np.float32, copy=False)
        states_buf = getattr(self._backend, "states_buf", None)
        if states_buf is not None:
            return self._tensor_to_numpy(states_buf).astype(np.float32, copy=False)
        return self._extract_policy_observation(raw_obs)

    def _extract_critic_observation_tensor(self, raw_obs: Any) -> torch.Tensor:
        if isinstance(raw_obs, Mapping):
            for key in (self._critic_obs_key, "critic", "states", "obs"):
                if key in raw_obs:
                    return self._to_backend_float_tensor(raw_obs[key])
        states_buf = getattr(self._backend, "states_buf", None)
        if states_buf is not None:
            return self._to_backend_float_tensor(states_buf)
        return self._extract_policy_observation_tensor(raw_obs)

    def _augment_policy_observation_with_command(
        self,
        policy_obs: np.ndarray,
        command: PasistCommand,
    ) -> np.ndarray:
        return self._augment_observation_with_command(
            observation=policy_obs,
            command=command,
            observation_name="policy observation",
        )

    def _augment_critic_observation_with_command(
        self,
        critic_obs: np.ndarray,
        command: PasistCommand,
    ) -> np.ndarray:
        return self._augment_observation_with_command(
            observation=critic_obs,
            command=command,
            observation_name="critic observation",
        )

    def _augment_policy_observation_with_command_tensor(
        self,
        policy_obs: torch.Tensor,
        command: PasistCommand,
    ) -> torch.Tensor:
        return self._augment_observation_with_command_tensor(
            observation=policy_obs,
            command=command,
            observation_name="policy observation",
        )

    def _augment_critic_observation_with_command_tensor(
        self,
        critic_obs: torch.Tensor,
        command: PasistCommand,
    ) -> torch.Tensor:
        return self._augment_observation_with_command_tensor(
            observation=critic_obs,
            command=command,
            observation_name="critic observation",
        )

    def _augment_observation_with_command(
        self,
        observation: np.ndarray,
        command: PasistCommand,
        observation_name: str,
    ) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        command_one_hot = np.asarray(command.one_hot, dtype=np.float32).reshape(-1)
        if command_one_hot.shape[0] != self.command_dim:
            raise ValueError(
                f"{observation_name} 拼接的 command one-hot 维度不正确："
                f"期望 {self.command_dim}，实际得到 {command_one_hot.shape[0]}"
            )
        if obs.ndim == 1:
            return np.concatenate([obs, command_one_hot], axis=0).astype(np.float32, copy=False)
        command_batch = np.broadcast_to(command_one_hot.reshape(1, -1), (obs.shape[0], command_one_hot.shape[0]))
        return np.concatenate([obs, command_batch.astype(np.float32, copy=False)], axis=-1).astype(np.float32, copy=False)

    def _augment_observation_with_command_tensor(
        self,
        observation: torch.Tensor,
        command: PasistCommand,
        observation_name: str,
    ) -> torch.Tensor:
        obs = self._to_backend_float_tensor(observation)
        command_one_hot = self._to_backend_float_tensor(command.one_hot).reshape(-1)
        if command_one_hot.shape[0] != self.command_dim:
            raise ValueError(
                f"{observation_name} 拼接的 command one-hot 维度不正确："
                f"期望 {self.command_dim}，实际得到 {command_one_hot.shape[0]}"
            )
        if obs.ndim == 1:
            return torch.cat([obs, command_one_hot], dim=0)
        command_batch = command_one_hot.reshape(1, -1).expand(obs.shape[0], -1)
        return torch.cat([obs, command_batch], dim=-1)

    def _build_step_info(
        self,
        policy_obs: np.ndarray,
        raw_obs: Any,
        raw_info: dict[str, Any],
        command: PasistCommand,
        env_reward: np.ndarray | None = None,
        terminated: np.ndarray | None = None,
        truncated: np.ndarray | None = None,
    ) -> dict[str, Any]:
        raw_critic_obs = self._extract_critic_observation(raw_obs)
        critic_obs = self._augment_critic_observation_with_command(raw_critic_obs, command)
        measured_velocity = self._extract_measured_velocity(raw_critic_obs)
        base_height = self._extract_base_height()
        base_pitch = self._extract_base_pitch()
        imitation_obs = self.extract_imitation_observation(policy_obs)
        extra = {
            "raw_obs": self._sanitize_to_host(raw_obs),
            "raw_info": self._sanitize_to_host(raw_info),
            "critic_obs": critic_obs.copy(),
            "imitation_obs": imitation_obs.copy(),
            "base_height": self._copy_or_scalar(base_height),
            "base_pitch": self._copy_or_scalar(base_pitch),
            "num_envs": self._num_envs,
            "backend_type": type(self._backend).__name__,
        }
        if env_reward is not None:
            extra["env_reward"] = env_reward.copy() if isinstance(env_reward, np.ndarray) else env_reward
        if terminated is not None:
            extra["terminated"] = terminated.copy() if isinstance(terminated, np.ndarray) else terminated
        if truncated is not None:
            extra["truncated"] = truncated.copy() if isinstance(truncated, np.ndarray) else truncated
        return self.build_info(
            observation=policy_obs,
            command=command,
            measured_velocity=measured_velocity,
            extra=extra,
        )

    def _build_step_info_tensor(
        self,
        policy_obs: torch.Tensor,
        raw_obs: Any,
        raw_info: dict[str, Any],
        command: PasistCommand,
        env_reward: torch.Tensor | None = None,
        terminated: torch.Tensor | None = None,
        truncated: torch.Tensor | None = None,
        lightweight: bool = False,
    ) -> dict[str, Any]:
        raw_critic_obs = self._extract_critic_observation_tensor(raw_obs)
        critic_obs = self._augment_critic_observation_with_command_tensor(raw_critic_obs, command)
        measured_velocity = self._extract_measured_velocity_tensor(raw_critic_obs)
        imitation_obs = self.extract_imitation_observation_tensor(policy_obs)
        info: dict[str, Any] = {
            "command": command,
            "skill_id": int(command.skill_id),
            "skill_name": command.skill_name,
            "target_pose": self.get_target_pose_tensor(command.skill_id),
            "imitation_obs": imitation_obs.detach().clone(),
            "measured_velocity": measured_velocity.detach().clone(),
            "critic_obs": critic_obs.detach().clone(),
            "num_envs": self._num_envs,
            "backend_type": type(self._backend).__name__,
        }
        if not lightweight:
            info["raw_obs"] = raw_obs
            info["raw_info"] = raw_info
            info["base_height"] = self._extract_base_height_tensor().detach().clone()
            info["base_pitch"] = self._extract_base_pitch_tensor().detach().clone()
        if env_reward is not None:
            info["env_reward"] = env_reward.detach().clone()
        if terminated is not None:
            info["terminated"] = terminated.detach().clone()
        if truncated is not None:
            info["truncated"] = truncated.detach().clone()
        return info

    def _apply_command_to_backend(self, command: PasistCommand) -> None:
        setter = getattr(self._backend, "set_pasist_command", None)
        if callable(setter):
            setter(command)
            return

        velocity_command = np.zeros((self._num_envs, 3), dtype=np.float32)
        velocity_command[:, 0] = float(command.velocity)

        command_tensor = self._to_backend_tensor(velocity_command)
        for attr_name in self._command_attr_candidates:
            target = getattr(self._backend, attr_name, None)
            if target is None:
                continue
            try:
                target[:, :3] = command_tensor[:, :3]
            except Exception:
                try:
                    target[:, :3] = velocity_command[:, :3]
                except Exception:
                    continue
            return

    def _extract_measured_velocity(self, critic_obs: np.ndarray) -> float | np.ndarray:
        obs = np.asarray(critic_obs, dtype=np.float32)
        if obs.ndim == 1:
            return float(obs[0]) if obs.size > 0 else 0.0
        if obs.ndim == 2:
            return obs[:, 0].astype(np.float32, copy=False)
        flattened = obs.reshape(-1, obs.shape[-1])
        return flattened[:, 0].astype(np.float32, copy=False)

    def _extract_measured_velocity_tensor(self, critic_obs: torch.Tensor) -> torch.Tensor:
        obs = self._to_backend_float_tensor(critic_obs)
        if obs.ndim == 1:
            return obs[0:1]
        if obs.ndim == 2:
            return obs[:, 0]
        flattened = obs.reshape(-1, obs.shape[-1])
        return flattened[:, 0]

    def _extract_joint_pos_rel(self, policy_obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(policy_obs, dtype=np.float32)
        start, end = self._joint_pos_rel_slice
        if obs.shape[-1] >= end:
            return obs[..., start:end].astype(np.float32, copy=False)

        dof_pos = self._get_backend_array(self._dof_pos_attr_candidates)
        default_dof_pos = self._get_backend_array(self._default_dof_pos_attr_candidates)
        if dof_pos is not None:
            dof_pos = np.asarray(dof_pos, dtype=np.float32)
            if default_dof_pos is None:
                return dof_pos[..., :12].astype(np.float32, copy=False)
            default = np.asarray(default_dof_pos, dtype=np.float32)
            return (dof_pos[..., :12] - default[..., :12]).astype(np.float32, copy=False)

        raise ValueError(
            f"policy observation 维度不足，无法提取 joint_pos_rel；"
            f"需要至少 {end} 维，实际为 {obs.shape[-1]}，且 backend 未提供 dof_pos。"
        )

    def _extract_joint_pos_rel_tensor(self, policy_obs: torch.Tensor) -> torch.Tensor:
        obs = self._to_backend_float_tensor(policy_obs)
        start, end = self._joint_pos_rel_slice
        if obs.shape[-1] >= end:
            return obs[..., start:end]

        dof_pos = self._get_backend_tensor(self._dof_pos_attr_candidates)
        default_dof_pos = self._get_backend_tensor(self._default_dof_pos_attr_candidates)
        if dof_pos is not None:
            if default_dof_pos is None:
                return dof_pos[..., :12]
            return dof_pos[..., :12] - default_dof_pos[..., :12]

        raise ValueError(
            f"policy observation 维度不足，无法提取 joint_pos_rel；"
            f"需要至少 {end} 维，实际为 {obs.shape[-1]}，且 backend 未提供 dof_pos。"
        )

    def _extract_base_height(self) -> float | np.ndarray:
        root_states = self._get_backend_array(self._root_state_attr_candidates)
        if root_states is None:
            zeros = np.zeros((self._num_envs,), dtype=np.float32)
            return float(zeros[0]) if self._num_envs == 1 else zeros
        height = np.asarray(root_states, dtype=np.float32)[:, 2]
        if self._num_envs == 1 and height.shape[0] == 1:
            return float(height[0])
        return height.astype(np.float32, copy=False)

    def _extract_base_height_tensor(self) -> torch.Tensor:
        root_states = self._get_backend_tensor(self._root_state_attr_candidates)
        if root_states is None:
            return torch.zeros((self._num_envs,), dtype=torch.float32, device=self._device)
        return root_states[:, 2].to(dtype=torch.float32)

    def _extract_base_pitch(self) -> float | np.ndarray:
        root_states = self._get_backend_array(self._root_state_attr_candidates)
        if root_states is None or np.asarray(root_states).shape[-1] < 7:
            zeros = np.zeros((self._num_envs,), dtype=np.float32)
            return float(zeros[0]) if self._num_envs == 1 else zeros
        quat = np.asarray(root_states, dtype=np.float32)[:, 3:7]
        pitch = self._pitch_from_quat(quat)
        if self._num_envs == 1 and pitch.shape[0] == 1:
            return float(pitch[0])
        return pitch.astype(np.float32, copy=False)

    def _extract_base_pitch_tensor(self) -> torch.Tensor:
        root_states = self._get_backend_tensor(self._root_state_attr_candidates)
        if root_states is None or root_states.shape[-1] < 7:
            return torch.zeros((self._num_envs,), dtype=torch.float32, device=self._device)
        quat = root_states[:, 3:7].to(dtype=torch.float32)
        return self._pitch_from_quat_tensor(quat)

    def _extract_current_skill_id(self) -> float | np.ndarray:
        active_command = self._current_command
        skill_id = float(self._default_skill_id if active_command is None else active_command.skill_id)
        if self._num_envs == 1:
            return skill_id
        return np.full((self._num_envs,), skill_id, dtype=np.float32)

    def _extract_current_skill_id_tensor(self) -> torch.Tensor:
        active_command = self._current_command
        skill_id = float(self._default_skill_id if active_command is None else active_command.skill_id)
        if self._num_envs == 1:
            return torch.tensor([skill_id], dtype=torch.float32, device=self._device)
        return torch.full((self._num_envs,), skill_id, dtype=torch.float32, device=self._device)

    def _get_backend_array(self, attr_candidates: Sequence[str]) -> np.ndarray | None:
        for attr_name in attr_candidates:
            value = getattr(self._backend, attr_name, None)
            if value is not None:
                return self._tensor_to_numpy(value).astype(np.float32, copy=False)
        return None

    def _get_backend_tensor(self, attr_candidates: Sequence[str]) -> torch.Tensor | None:
        for attr_name in attr_candidates:
            value = getattr(self._backend, attr_name, None)
            if value is not None:
                return self._to_backend_float_tensor(value)
        return None

    def _pitch_from_quat(self, quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float32)
        if self._root_quat_order == "xyzw":
            x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        else:
            w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        return np.arcsin(sinp).astype(np.float32, copy=False)

    def _pitch_from_quat_tensor(self, quat: torch.Tensor) -> torch.Tensor:
        quat_tensor = self._to_backend_float_tensor(quat)
        if quat_tensor.ndim == 1:
            quat_tensor = quat_tensor.reshape(1, 4)
        if self._root_quat_order == "xyzw":
            x, y, z, w = quat_tensor[:, 0], quat_tensor[:, 1], quat_tensor[:, 2], quat_tensor[:, 3]
        else:
            w, x, y, z = quat_tensor[:, 0], quat_tensor[:, 1], quat_tensor[:, 2], quat_tensor[:, 3]
        sinp = 2.0 * (w * y - z * x)
        sinp = torch.clamp(sinp, -1.0, 1.0)
        return torch.asin(sinp).to(dtype=torch.float32)

    def _to_action_tensor(self, action: torch.Tensor | np.ndarray):
        if torch.is_tensor(action):
            action_tensor = action.to(device=self._device, dtype=torch.float32)
            if action_tensor.ndim == 1:
                if self._num_envs != 1:
                    action_tensor = action_tensor.reshape(1, self._action_dim).expand(self._num_envs, -1)
                else:
                    action_tensor = action_tensor.reshape(1, self._action_dim)
            elif action_tensor.ndim != 2:
                raise ValueError(f"action 维度不合法，期望 1D 或 2D，实际为 {action_tensor.ndim}D")
            if tuple(action_tensor.shape) != (self._num_envs, self._action_dim):
                raise ValueError(
                    f"action shape 不匹配，期望 {(self._num_envs, self._action_dim)}，实际为 {tuple(action_tensor.shape)}"
                )
            return action_tensor

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
        return self._to_backend_tensor(action_array)

    def _to_backend_tensor(self, value: np.ndarray):
        try:
            return torch.as_tensor(value, dtype=torch.float32, device=self._device)
        except Exception:
            return np.asarray(value, dtype=np.float32)

    def _to_backend_float_tensor(self, value: Any) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(device=self._device, dtype=torch.float32)
        return torch.as_tensor(value, dtype=torch.float32, device=self._device)

    def _to_backend_bool_tensor(self, value: Any) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(device=self._device, dtype=torch.bool)
        return torch.as_tensor(value, dtype=torch.bool, device=self._device)

    def _tensor_to_numpy(self, value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return np.asarray(value)

    def _sanitize_to_host(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: self._sanitize_to_host(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self._sanitize_to_host(item) for item in value)
        if isinstance(value, list):
            return [self._sanitize_to_host(item) for item in value]
        if hasattr(value, "detach") or hasattr(value, "cpu"):
            return self._tensor_to_numpy(value)
        return value

    def _copy_or_scalar(self, value: Any) -> Any:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 0:
            return float(array)
        return array.copy()

    def _maybe_squeeze_obs(self, observation: np.ndarray) -> np.ndarray:
        if self._num_envs == 1 and observation.ndim >= 2 and observation.shape[0] == 1:
            return observation[0]
        return observation

    def _maybe_squeeze_tensor_obs(self, observation: torch.Tensor) -> torch.Tensor:
        if self._num_envs == 1 and observation.ndim >= 2 and observation.shape[0] == 1:
            return observation[0]
        return observation

    def _maybe_squeeze_scalar(self, value: Any) -> Any:
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == 1:
            scalar = value[0]
            if isinstance(scalar, np.generic):
                return scalar.item()
            return scalar
        return value

    def _maybe_squeeze_tensor_scalar(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim >= 1 and value.shape[0] == 1:
            return value[0]
        return value

    def _maybe_squeeze_info(self, info: dict[str, Any]) -> dict[str, Any]:
        if self._num_envs != 1:
            return info
        squeezed: dict[str, Any] = {}
        for key, value in info.items():
            if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == 1:
                squeezed[key] = value[0]
            else:
                squeezed[key] = value
        return squeezed

    def _maybe_squeeze_tensor_info(self, info: dict[str, Any]) -> dict[str, Any]:
        if self._num_envs != 1:
            return info
        squeezed: dict[str, Any] = {}
        for key, value in info.items():
            if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == 1:
                squeezed[key] = value[0]
            else:
                squeezed[key] = value
        return squeezed
