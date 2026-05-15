from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.distributions import Normal

from envs.BasePasistEnv import BasePasistEnv, PasistCommand
from rewards import (
    compute_mean_sil_dtw,
    compute_regularization_reward,
    compute_reward_terms,
    compute_sil_confidence_weight,
    compute_sil_weight,
    compute_sil_warmup_weight,
    compute_task_reward,
)
from rewards.torch_reward import (
    compute_regularization_reward_torch,
    compute_reward_terms_torch,
    compute_task_reward_torch,
)
from rl.actor_critic_new import MLP
from rl.rollout_buffer import RolloutBuffer
from sil.discriminator import SILDiscriminator, load_discriminator_config
from sil.sil_buffer import SILBuffer
from sil.skill_selector import SkillSelector
from sil.trajectory_selector import TrajectorySelector


@dataclass
class PPOTrainerConfig:
    """
    PPOTrainer 的核心超参数配置。

    这一版配置刻意保持“够用但不臃肿”：
    - 前半部分是 PPO 本体超参数
    - 中间是 reward 组合相关超参数
    - 后半部分是 PASIST / SIL 的可选开关

    如果后面你要把它搬到 YAML，再逐步拆出去会更自然。
    """
    
    # rollout 和 PPO 更新相关配置
    num_steps: int = 128
    ppo_epochs: int = 4
    mini_batch_size: int = 256
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    actor_lr: float = 3.0e-4
    critic_lr: float = 1.0e-3
    max_grad_norm: float = 1.0
    value_loss_coef: float = 0.25
    entropy_coef: float = 0.0
    device: str = "cpu"

    # policy / value 网络结构
    actor_hidden_dims: tuple[int, ...] = (256, 256)
    critic_hidden_dims: tuple[int, ...] = (256, 256)
    actor_activation: str = "relu"
    critic_activation: str = "relu"
    init_log_std: float = -0.5

    # task reward 相关超参数
    pose_weight: float = 1.0
    velocity_weight: float = 0.0
    command_weight: float = 0.0
    pose_sigma: float = 1.0
    velocity_sigma: float = 0.25
    yaw_weight: float = 0.0
    yaw_sigma: float = 0.25
    lin_vel_z_weight: float = 0.0
    ang_vel_xy_weight: float = 0.0
    flat_orientation_weight: float = 0.0
    height_weight: float = 0.0
    height_sigma: float = 0.05

    # regularization reward 相关超参数
    action_weight: float = 1.0e-3
    smoothness_weight: float = 1.0e-3
    stability_weight: float = 0.0
    joint_acceleration_weight: float = 0.0
    roll_pitch_rate_weight: float = 0.0
    yaw_rate_weight: float = 0.0
    lateral_velocity_weight: float = 0.0

    # total reward 动态权重相关超参数
    sigma_t: float = 0.5
    sigma_sil: float = 0.5
    sil_reward_positive_margin: float = 0.1
    sil_weight_dtw_decay_rate: float = 10.0
    sil_weight_confidence_min_margin: float = 0.2
    sil_weight_confidence_max_margin: float = 0.6
    sil_weight_warmup_trajectories: int = 20
    sil_weight_cap: float = 0.4
    omega_r: float = 1.0

    # PASIST / SIL 可选开关
    enable_sil: bool = True
    use_skill_selector: bool = False
    discriminator_config_path: str = "configs/training/discriminator.yaml"
    discriminator_batch_size: int = 256
    min_sil_buffer_trajectories_for_discriminator: int = 8
    discriminator_updates_per_iteration: int = 1
    discriminator_max_updates_per_iteration: int = 4
    discriminator_min_updates_per_iteration: int = 1
    discriminator_adaptive_updates: bool = True
    sil_buffer_capacity_per_skill: int = 32
    trajectory_dtw_weight: float = 0.1
    trajectory_reference_length_ratio: float = 0.5
    trajectory_normalize_dtw: bool = True
    trajectory_dtw_feature_slices: list[tuple[int, int]] | None = None
    trajectory_task_reward_normalization_length: float = 128.0
    sil_update_interval: int = 1
    sil_use_completed_episodes_only: bool = True
    use_tensor_env_api: bool = False
    use_torch_reward_stack: bool = False
    use_lightweight_tensor_info: bool = False


class GaussianPolicy(nn.Module):
    """
    连续动作 PPO 使用的对角高斯策略网络。

    输入:
    - observation，shape = [batch_size, obs_dim]

    输出:
    - action mean，shape = [batch_size, action_dim]
    - log_std 是可学习的全局参数，shape = [action_dim]
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims,
        init_log_std: float = -0.5,
        hidden_activation: str = "relu",
    ) -> None:
        super().__init__()
        self.mean_net = MLP(
            input_dim=obs_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
            output_activation=None,
            hidden_activation=hidden_activation,
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std), dtype=torch.float32))

    def distribution(self, observations: torch.Tensor) -> Normal:
        """返回当前观测下的高斯动作分布。"""
        mean = self.mean_net(observations)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def sample(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        从策略分布中采样动作。

        返回:
        - actions: shape = [batch_size, action_dim]
        - log_probs: shape = [batch_size, 1]
        """
        distribution = self.distribution(observations)
        actions = distribution.sample()
        log_probs = distribution.log_prob(actions).sum(dim=-1, keepdim=True)
        return actions, log_probs

    def log_prob_and_entropy(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算给定动作的 log_prob 和策略熵。

        返回:
        - log_probs: shape = [batch_size, 1]
        - entropy: 标量张量
        """
        distribution = self.distribution(observations)
        log_probs = distribution.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = distribution.entropy().sum(dim=-1).mean()
        return log_probs, entropy


class ValueFunction(nn.Module):
    """
    PPO 中的状态价值网络 V(s)。
    """

    def __init__(self, obs_dim: int, hidden_dims, hidden_activation: str = "relu") -> None:
        super().__init__()
        self.value_net = MLP(
            input_dim=obs_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            output_activation=None,
            hidden_activation=hidden_activation,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """输入 observations，输出 shape = [batch_size, 1] 的状态价值。"""
        return self.value_net(observations)


class PPOTrainer:
    """
    面向当前 PASIST 项目骨架的 PPO trainer。

    这一版 trainer 的设计重点：
    1. 先把环境、reward、rollout buffer 和连续动作 PPO 串起来
    2. 给 PASIST 的 SIL / trajectory selector / skill selector 留出清晰接口
    3. 默认允许你先关掉 SIL，仅运行 task + regularization 版本

    当前最稳的使用方式：
    - 先用单技能 `walk`
    - `num_envs = 1` 或少量并行环境
    - `enable_sil = False` 先把主链跑通
    - 再逐步打开 trajectory selector / discriminator / skill selector
    """

    def __init__(
        self,
        env: BasePasistEnv,
        config: PPOTrainerConfig | None = None,
        skill_selector: SkillSelector | None = None,
        sil_buffer: SILBuffer | None = None,
        trajectory_selector: TrajectorySelector | None = None,
        discriminator: SILDiscriminator | None = None,
        discriminator_optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.env = env
        self.config = PPOTrainerConfig() if config is None else config
        self.device = self._resolve_device(self.config.device)
        self.num_envs = int(getattr(env, "num_envs", 1))
        self.policy_obs_dim = int(env.obs_dim)
        self.critic_obs_dim = int(getattr(env, "critic_obs_dim", env.obs_dim))

        self.policy = GaussianPolicy(
            obs_dim=self.policy_obs_dim,
            action_dim=env.action_dim,
            hidden_dims=self.config.actor_hidden_dims,
            init_log_std=self.config.init_log_std,
            hidden_activation=self.config.actor_activation,
        ).to(self.device)
        self.value_function = ValueFunction(
            obs_dim=self.critic_obs_dim,
            hidden_dims=self.config.critic_hidden_dims,
            hidden_activation=self.config.critic_activation,
        ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.value_function.parameters(), lr=self.config.critic_lr)

        self.rollout_buffer = RolloutBuffer(
            num_steps=self.config.num_steps,
            num_envs=self.num_envs,
            obs_dim=self.policy_obs_dim,
            critic_obs_dim=self.critic_obs_dim,
            act_dim=env.action_dim,
            device=self.device,
            imitation_obs_dim=env.imitation_obs_dim,
            command_dim=env.command_dim,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        self.skill_selector = skill_selector
        if self.skill_selector is None and self.config.use_skill_selector:
            self.skill_selector = SkillSelector(
                num_skills=env.num_skills,
                velocity_range=env.velocity_range,
            )

        discriminator_config: dict[str, Any] = {}
        discriminator_config_path = Path(self.config.discriminator_config_path)
        if discriminator_config_path.exists():
            discriminator_config = load_discriminator_config(discriminator_config_path)

        trajectory_selector_cfg = discriminator_config.get("trajectory_selector", {})
        if not isinstance(trajectory_selector_cfg, dict):
            trajectory_selector_cfg = {}

        self.sil_buffer = sil_buffer or SILBuffer(
            capacity_per_skill=self.config.sil_buffer_capacity_per_skill,
        )
        self.trajectory_selector = trajectory_selector or TrajectorySelector(
            dtw_weight=self.config.trajectory_dtw_weight,
            reference_length_ratio=self.config.trajectory_reference_length_ratio,
            normalize_dtw=self.config.trajectory_normalize_dtw,
            dtw_feature_slices=self._normalize_feature_slices(self.config.trajectory_dtw_feature_slices),
            task_reward_normalization_length=self.config.trajectory_task_reward_normalization_length,
            min_task_return=float(trajectory_selector_cfg.get("min_task_return", -np.inf)),
            max_dtw_distance=float(trajectory_selector_cfg.get("max_dtw_distance", np.inf)),
        )

        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        if self.config.enable_sil and self.discriminator is None:
            if discriminator_config_path.exists():
                self.discriminator = SILDiscriminator.from_yaml(
                    input_dim=self._infer_discriminator_input_dim(),
                    config_path=discriminator_config_path,
                ).to(self.device)
                self.discriminator_optimizer = self.discriminator.build_optimizer(config_path=discriminator_config_path)

        # 判别器当前使用 [base_height, joint_pos_rel(12), command_onehot] 的单帧条件输入。
        # 其中 joint_pos_rel 单独做固定尺度归一化，减少不同关节摆幅带来的量级差异。
        self._discriminator_joint_pos_scale = 0.25

        # 自适应判别器更新次数：记录上一轮 expert-policy 分离度
        self._last_disc_separation: float = 0.0
        self._last_disc_expert_score: float = 0.0
        self._last_disc_policy_score: float = 0.0

        self._current_obs: np.ndarray | None = None
        self._current_info: dict[str, Any] | None = None
        self._current_command: PasistCommand | None = None
        self._previous_action = np.zeros((self.num_envs, env.action_dim), dtype=np.float32)
        self._episode_ids = np.zeros(self.num_envs, dtype=np.int64)
        self.total_env_steps = 0

    def train(
        self,
        num_iterations: int,
        seed: int | None = None,
        on_iteration_end: Callable[[dict[str, float]], None] | None = None,
    ) -> list[dict[str, float]]:
        """
        连续运行若干个 PPO iteration。

        输入:
        - `num_iterations`: 训练多少个 rollout + update 周期
        - `seed`: 可选；仅在第一次 reset 时传给环境
        - `on_iteration_end`:
          可选回调。若提供，则每个 iteration 结束后都会收到当前统计字典，
          适合在训练过程中实时写日志、更新 tensorboard 或保存 checkpoint。

        输出:
        - `list[dict[str, float]]`
          每个 iteration 一条日志字典
        """
        history: list[dict[str, float]] = []
        for iteration in range(int(num_iterations)):
            stats = self.train_iteration(
                seed=seed if iteration == 0 else None,
                iteration_index=iteration,
            )
            stats["iteration"] = float(iteration)
            history.append(stats)
            if on_iteration_end is not None:
                on_iteration_end(dict(stats))
        return history

    def train_iteration(
        self,
        seed: int | None = None,
        iteration_index: int | None = None,
    ) -> dict[str, float]:
        """
        执行一次完整训练迭代。

        迭代内容：
        1. 收集 rollout
        2. 进行 PPO 更新
        3. 用 rollout 轨迹更新 SIL buffer
        4. 可选：更新 discriminator
        5. 清空 rollout buffer
        """
        iteration_start_time = time.perf_counter()
        if self._current_obs is None:
            self._reset_env_with_new_command(seed=seed)

        collection_start_time = time.perf_counter()
        rollout_stats = self.collect_rollout()
        collection_time = time.perf_counter() - collection_start_time

        learning_start_time = time.perf_counter()
        ppo_start_time = learning_start_time
        ppo_stats = self.update_policy()
        ppo_time = time.perf_counter() - ppo_start_time

        sil_start_time = time.perf_counter()
        sil_stats = self.update_sil_components(iteration_index=iteration_index)
        sil_time = time.perf_counter() - sil_start_time
        learning_time = time.perf_counter() - learning_start_time

        iteration_time = time.perf_counter() - iteration_start_time
        rollout_steps = int(self.config.num_steps * self.num_envs)
        self.total_env_steps += rollout_steps

        self.rollout_buffer.clear()

        merged = {}
        merged.update(rollout_stats)
        merged.update(ppo_stats)
        merged.update(sil_stats)
        merged.update(
            {
                "iteration_collection_time_sec": float(collection_time),
                "iteration_learning_time_sec": float(learning_time),
                "iteration_learning_ppo_time_sec": float(ppo_time),
                "iteration_learning_sil_time_sec": float(sil_time),
                "iteration_total_time_sec": float(iteration_time),
                "iteration_steps_per_sec": float(rollout_steps / max(iteration_time, 1.0e-8)),
                "iteration_env_steps": float(rollout_steps),
                "total_env_steps": float(self.total_env_steps),
                "mean_action_noise_std": float(torch.exp(self.policy.log_std).mean().item()),
            }
        )
        return merged

    def collect_rollout(self) -> dict[str, float]:
        """
        与环境交互 `num_steps` 步，并把数据写入 rollout buffer。

        当前奖励链：
        - task reward: 由 `rewards/task_reward.py` 计算
        - regularization reward: 由 `rewards/regularization_reward.py` 计算
        - sil reward: 如果开启 SIL 且判别器可用，则额外计算
        - total reward: 由 `rewards/total_reward.py` 动态混合
        """
        reward_total_sum = 0.0
        reward_task_sum = 0.0
        reward_reg_sum = 0.0
        reward_sil_sum = 0.0
        omega_t_sum = 0.0
        omega_sil_sum = 0.0
        omega_sil_dtw_sum = 0.0
        omega_sil_confidence_sum = 0.0
        omega_sil_warmup_sum = 0.0
        done_count = 0.0
        use_tensor_env_api = self._use_tensor_env_api()
        use_torch_reward_stack = bool(self.config.use_torch_reward_stack)

        for _ in range(self.config.num_steps):
            if use_tensor_env_api:
                obs_tensor = self._ensure_batch_observation_tensor(self._current_obs)
                critic_obs_tensor = self._ensure_batch_critic_observation_tensor(
                    self._current_info.get("critic_obs") if self._current_info is not None else None,
                    fallback=obs_tensor,
                )
            else:
                obs_batch = self._ensure_batch_observation(self._current_obs)
                obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
                critic_obs_batch = self._ensure_batch_critic_observation(
                    self._current_info.get("critic_obs") if self._current_info is not None else None,
                    fallback=obs_batch,
                )
                critic_obs_tensor = torch.as_tensor(critic_obs_batch, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                actions_tensor, log_probs_tensor = self.policy.sample(obs_tensor)
                values_tensor = self.value_function(critic_obs_tensor)

            actions_np: np.ndarray | None = None
            if use_tensor_env_api:
                next_obs, _, terminated, truncated, info = self.env.step_tensor(
                    actions_tensor if self.num_envs > 1 else actions_tensor[0],
                    lightweight_info=self.config.use_lightweight_tensor_info,
                )
                next_obs_tensor = self._ensure_batch_observation_tensor(next_obs)
                self._ensure_batch_critic_observation_tensor(info.get("critic_obs"), fallback=next_obs_tensor)
                reward_input_action: Any = actions_tensor
                reward_input_obs: Any = next_obs_tensor
            else:
                actions_np = actions_tensor.detach().cpu().numpy().astype(np.float32)
                next_obs, _, terminated, truncated, info = self.env.step(actions_np if self.num_envs > 1 else actions_np[0])
                next_obs_batch = self._ensure_batch_observation(next_obs)
                self._ensure_batch_critic_observation(info.get("critic_obs"), fallback=next_obs_batch)
                reward_input_action = actions_np
                reward_input_obs = next_obs_batch
                next_obs_tensor = torch.as_tensor(next_obs_batch, dtype=torch.float32, device=self.device)

            if use_torch_reward_stack:
                reward_terms = self._compute_step_reward_terms_torch(
                    action=reward_input_action,
                    next_observation=reward_input_obs,
                    info=info,
                )
            else:
                if actions_np is None:
                    actions_np = actions_tensor.detach().cpu().numpy().astype(np.float32)
                reward_terms = self._compute_step_reward_terms(
                    action=actions_np,
                    next_observation=self._ensure_batch_observation(next_obs_tensor),
                    info=info,
                )

            done_mask = torch.logical_or(
                self._ensure_batch_bool_tensor(terminated),
                self._ensure_batch_bool_tensor(truncated),
            )
            reward_total = self._ensure_batch_column_tensor(reward_terms["total_reward"])
            reward_task = self._ensure_batch_column_tensor(reward_terms["task_reward"])
            reward_reg = self._ensure_batch_column_tensor(reward_terms["regularization_reward"])
            reward_sil = self._ensure_batch_column_tensor(reward_terms["sil_reward"])

            active_command = self._current_command or self._sample_training_command()
            self.rollout_buffer.add(
                obs=obs_tensor.detach(),
                critic_obs=critic_obs_tensor.detach(),
                action=actions_tensor.detach(),
                log_prob=log_probs_tensor.detach(),
                value=values_tensor.detach(),
                reward_total=reward_total,
                reward_task=reward_task,
                reward_reg=reward_reg,
                reward_sil=reward_sil,
                done=self._ensure_batch_column_tensor(done_mask.to(dtype=torch.float32)),
                next_obs=next_obs_tensor.detach(),
                skill_id=self._build_skill_id_column_tensor(active_command.skill_id),
                imitation_obs=self._ensure_batch_matrix_tensor(info["imitation_obs"], self.env.imitation_obs_dim),
                episode_id=torch.as_tensor(self._episode_ids.reshape(self.num_envs, 1), dtype=torch.long, device=self.device),
                terminated=self._ensure_batch_column_tensor(self._ensure_batch_bool_tensor(terminated).to(dtype=torch.float32)),
                truncated=self._ensure_batch_column_tensor(self._ensure_batch_bool_tensor(truncated).to(dtype=torch.float32)),
                command_speed=self._build_command_speed_column_tensor(active_command.velocity),
                command_onehot=self._build_command_onehot_batch_tensor(active_command.one_hot),
            )

            reward_total_sum += self._mean_scalar(reward_terms["total_reward"])
            reward_task_sum += self._mean_scalar(reward_terms["task_reward"])
            reward_reg_sum += self._mean_scalar(reward_terms["regularization_reward"])
            reward_sil_sum += self._mean_scalar(reward_terms["sil_reward"])
            omega_t_sum += self._mean_scalar(reward_terms["omega_t"])
            omega_sil_sum += self._mean_scalar(reward_terms["omega_sil"])
            omega_sil_dtw_sum += self._mean_scalar(reward_terms["omega_sil_dtw"])
            omega_sil_confidence_sum += self._mean_scalar(reward_terms["omega_sil_confidence"])
            omega_sil_warmup_sum += self._mean_scalar(reward_terms["omega_sil_warmup"])
            done_count += float(done_mask.to(dtype=torch.float32).sum().item())

            self._current_obs = next_obs_tensor.detach() if use_tensor_env_api else self._ensure_batch_observation(next_obs_tensor)
            self._current_info = info
            self._previous_action = actions_tensor.detach().clone() if (use_tensor_env_api or use_torch_reward_stack) else actions_np.copy()

            # 当前版本最稳的 episode 切换逻辑：
            # - 单环境时，done 后显式 reset，并为下一个 episode 采样新 command
            # - 多环境时，底层 Isaac Lab 已经自动 reset，对单技能版本保持当前 command 不变
            if bool(done_mask.any().item()):
                done_indices = torch.nonzero(done_mask, as_tuple=False).reshape(-1).detach().cpu().numpy()
                self._episode_ids[done_indices] += 1

                if self.num_envs == 1:
                    self._reset_env_with_new_command(seed=None)
                else:
                    if torch.is_tensor(self._previous_action):
                        index_tensor = torch.as_tensor(done_indices, dtype=torch.long, device=self.device)
                        self._previous_action[index_tensor] = 0.0
                    else:
                        self._previous_action[done_indices] = 0.0

        return {
            "rollout_reward_total_mean": reward_total_sum / max(self.config.num_steps, 1),
            "rollout_reward_task_mean": reward_task_sum / max(self.config.num_steps, 1),
            "rollout_reward_reg_mean": reward_reg_sum / max(self.config.num_steps, 1),
            "rollout_reward_sil_mean": reward_sil_sum / max(self.config.num_steps, 1),
            "rollout_omega_t_mean": omega_t_sum / max(self.config.num_steps, 1),
            "rollout_omega_sil_mean": omega_sil_sum / max(self.config.num_steps, 1),
            "rollout_omega_sil_dtw_mean": omega_sil_dtw_sum / max(self.config.num_steps, 1),
            "rollout_omega_sil_confidence_mean": omega_sil_confidence_sum / max(self.config.num_steps, 1),
            "rollout_omega_sil_warmup_mean": omega_sil_warmup_sum / max(self.config.num_steps, 1),
            "rollout_done_count": done_count,
        }

    def update_policy(self) -> dict[str, float]:
        """
        使用当前 rollout buffer 做 PPO 更新。
        """
        if self._current_obs is None:
            return {}

        last_obs_tensor = torch.as_tensor(
            self._ensure_batch_critic_observation(
                self._current_info.get("critic_obs") if self._current_info is not None else None,
                fallback=self._current_obs,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            last_value = self.value_function(last_obs_tensor)

        self.rollout_buffer.compute_returns_and_advantages(last_value=last_value)

        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0
        approx_kl_sum = 0.0
        num_updates = 0

        for _ in range(self.config.ppo_epochs):
            for minibatch in self.rollout_buffer.get_minibatches(self.config.mini_batch_size, shuffle=True):
                observations = minibatch["obs"]
                critic_observations = minibatch["critic_obs"]
                actions = minibatch["actions"]
                old_log_probs = minibatch["log_probs"]
                returns = minibatch["returns"]
                advantages = minibatch["advantages"]

                new_log_probs, entropy = self.policy.log_prob_and_entropy(observations, actions)
                ratios = torch.exp(new_log_probs - old_log_probs)
                surrogate_1 = ratios * advantages
                surrogate_2 = torch.clamp(
                    ratios,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                ) * advantages
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                values = self.value_function(critic_observations)
                value_loss = F.mse_loss(values, returns)

                self.actor_optimizer.zero_grad()
                (policy_loss - self.config.entropy_coef * entropy).backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                (self.config.value_loss_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.value_function.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()

                approx_kl = (old_log_probs - new_log_probs).mean()
                policy_loss_sum += float(policy_loss.item())
                value_loss_sum += float(value_loss.item())
                entropy_sum += float(entropy.item())
                approx_kl_sum += float(approx_kl.item())
                num_updates += 1

        if num_updates == 0:
            return {}

        return {
            "ppo_policy_loss": policy_loss_sum / num_updates,
            "ppo_value_loss": value_loss_sum / num_updates,
            "ppo_entropy": entropy_sum / num_updates,
            "ppo_approx_kl": approx_kl_sum / num_updates,
        }

    def update_sil_components(self, iteration_index: int | None = None) -> dict[str, float]:
        """
        使用本轮 rollout 更新 PASIST 的轨迹筛选和判别器模块。

        当前流程：
        1. 从 rollout buffer 中提 episode
        2. 用 trajectory selector 计算 DTW 和 assessment score
        3. 接纳高质量轨迹到 SIL buffer
        4. 如果开启了 discriminator，则再做若干次更新
        """
        sil_update_interval = max(int(self.config.sil_update_interval), 1)
        stats = {
            "sil_update_interval": float(sil_update_interval),
            "sil_update_triggered": float(0.0),
        }

        if not self.config.enable_sil and self.skill_selector is None:
            return stats

        if iteration_index is not None and sil_update_interval > 1 and iteration_index % sil_update_interval != 0:
            summary = self.sil_buffer.summary()
            stats.update(
                {
                    "trajectory_candidate_count": float(0.0),
                    "trajectory_evaluated_count": float(0.0),
                    "trajectory_accepted_count": float(0.0),
                    "sil_buffer_num_trajectories": float(len(self.sil_buffer)),
                    "sil_buffer_mean_dtw": float(compute_mean_sil_dtw(summary)),
                }
            )
            return stats

        episodes = self.rollout_buffer.extract_episodes_tensor()
        candidate_episodes = self._select_sil_candidate_episodes(episodes)
        accepted_count = 0
        evaluated_count = 0

        for episode in candidate_episodes:
            if self._episode_is_tensor_dict(episode):
                skill_id = self._extract_episode_skill_id(episode)
                target_pose = self.env.get_target_pose(skill_id)
                evaluation = self.trajectory_selector.evaluate_episode_tensor(
                    episode=episode,
                    target_pose=target_pose,
                    device=self.device,
                )
            else:
                trajectory = self.trajectory_selector.episode_from_buffer_dict(episode)
                target_pose = self.env.get_target_pose(trajectory.skill_id)
                evaluation = self.trajectory_selector.evaluate(trajectory=trajectory, target_pose=target_pose)
                skill_id = trajectory.skill_id
            evaluated_count += 1

            if self.skill_selector is not None:
                self.skill_selector.update(skill_id=skill_id, task_reward=evaluation.task_return)

            if evaluation.accepted:
                accepted_count += 1
                metadata = {
                    "dtw_distance": evaluation.dtw_distance,
                    "task_return": evaluation.task_return,
                    "trajectory_length": evaluation.trajectory_length,
                }
                if self._episode_is_tensor_dict(episode):
                    self.sil_buffer.add_episode_tensor(
                        episode=episode,
                        assessment_score=evaluation.assessment_score,
                        skill_id=skill_id,
                        metadata=metadata,
                    )
                else:
                    self.sil_buffer.add_episode(
                        episode=episode,
                        assessment_score=evaluation.assessment_score,
                        skill_id=skill_id,
                        metadata=metadata,
                    )

        discriminator_stats = self.update_discriminator()
        summary = self.sil_buffer.summary()

        stats.update(
            {
                "sil_update_triggered": float(1.0),
                "trajectory_candidate_count": float(len(candidate_episodes)),
                "trajectory_evaluated_count": float(evaluated_count),
                "trajectory_accepted_count": float(accepted_count),
                "sil_buffer_num_trajectories": float(len(self.sil_buffer)),
                "sil_buffer_mean_dtw": float(compute_mean_sil_dtw(summary)),
            }
        )
        stats.update(self._summarize_episode_metrics(candidate_episodes if candidate_episodes else episodes))
        stats.update(discriminator_stats)
        return stats

    def _select_sil_candidate_episodes(self, episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        选择本轮参与 DTW / SIL 评估的 episode 集合。

        默认优先只使用完整结束的 episode：
        - 这通常能显著减少 DTW 数量
        - 也避免把大量不完整片段塞进 trajectory selector
        - 如果当前一轮没有完整 episode，则回退到原始片段，避免完全无数据
        """
        if not episodes:
            return []
        if not self.config.sil_use_completed_episodes_only:
            return episodes

        completed_episodes = []
        for episode in episodes:
            if not self._episode_finished(episode):
                continue
            completed_episodes.append(episode)
        return completed_episodes if completed_episodes else episodes

    def _episode_finished(self, episode: dict[str, Any]) -> bool:
        """判断一个 episode 片段是否以 done 结束。"""
        done_sequence = episode.get("dones")
        if done_sequence is None:
            return False
        if torch.is_tensor(done_sequence):
            if done_sequence.numel() == 0:
                return False
            done_flag = float(done_sequence.reshape(-1)[-1].detach().to(dtype=torch.float32).item())
            return done_flag > 0.5
        if not done_sequence:
            return False
        done_flag = float(self._to_numpy(done_sequence[-1]).reshape(-1)[0])
        return done_flag > 0.5

    def _summarize_episode_metrics(self, episodes: list[dict[str, Any]]) -> dict[str, float]:
        """
        汇总 rollout 中 episode 级别的奖励与长度统计。

        优先使用本轮已完成的 episode；如果一轮内没有完整结束的 episode，
        则退化为使用当前收集到的片段，避免控制台输出完全空白。
        """
        if not episodes:
            return {}

        completed_episodes = []
        for episode in episodes:
            if self._episode_finished(episode):
                completed_episodes.append(episode)

        target_episodes = completed_episodes if completed_episodes else episodes
        if not target_episodes:
            return {}

        total_returns = []
        task_returns = []
        reg_returns = []
        sil_returns = []
        lengths = []

        for episode in target_episodes:
            if self._episode_is_tensor_dict(episode):
                reward_total = self._episode_reward_tensor(episode, "reward_total")
                reward_task = self._episode_reward_tensor(episode, "reward_task")
                reward_reg = self._episode_reward_tensor(episode, "reward_reg")
                reward_sil = self._episode_reward_tensor(episode, "reward_sil")

                total_returns.append(float(reward_total.sum().item()))
                task_returns.append(float(reward_task.sum().item()))
                reg_returns.append(float(reward_reg.sum().item()))
                sil_returns.append(float(reward_sil.sum().item()))
                lengths.append(float(reward_total.shape[0]))
            else:
                reward_total = np.asarray(
                    [self._to_numpy(item, dtype=np.float32).reshape(-1)[0] for item in episode["reward_total"]],
                    dtype=np.float32,
                )
                reward_task = np.asarray(
                    [self._to_numpy(item, dtype=np.float32).reshape(-1)[0] for item in episode["reward_task"]],
                    dtype=np.float32,
                )
                reward_reg = np.asarray(
                    [self._to_numpy(item, dtype=np.float32).reshape(-1)[0] for item in episode["reward_reg"]],
                    dtype=np.float32,
                )
                reward_sil = np.asarray(
                    [self._to_numpy(item, dtype=np.float32).reshape(-1)[0] for item in episode["reward_sil"]],
                    dtype=np.float32,
                )

                total_returns.append(float(reward_total.sum()))
                task_returns.append(float(reward_task.sum()))
                reg_returns.append(float(reward_reg.sum()))
                sil_returns.append(float(reward_sil.sum()))
                lengths.append(float(len(reward_total)))

        return {
            "episode_reward_mean": float(np.mean(total_returns)),
            "episode_reward_task_mean": float(np.mean(task_returns)),
            "episode_reward_reg_mean": float(np.mean(reg_returns)),
            "episode_reward_sil_mean": float(np.mean(sil_returns)),
            "episode_length_mean": float(np.mean(lengths)),
            "episode_count": float(len(target_episodes)),
            "completed_episode_count": float(len(completed_episodes)),
        }

    def _compute_discriminator_update_count(self) -> int:
        """根据上一轮判别器分离度动态计算本轮应做的更新次数。

        规则（基于 20260512-153855 日志中的坍缩模式）：
        - separation < 0.15（严重坍缩）→ 用最大次数
        - separation > 0.60（判别器过强）→ 用最小次数
        - 中间区域线性插值
        """
        separation = abs(self._last_disc_separation)
        max_updates = int(self.config.discriminator_max_updates_per_iteration)
        min_updates = int(self.config.discriminator_min_updates_per_iteration)
        if max_updates <= min_updates:
            return max(max_updates, 1)

        collapsed_threshold = 0.15
        strong_threshold = 0.60

        if separation <= collapsed_threshold:
            return max_updates
        if separation >= strong_threshold:
            return min_updates
        ratio = (separation - collapsed_threshold) / (strong_threshold - collapsed_threshold)
        return max(min_updates, int(round(max_updates - ratio * (max_updates - min_updates))))

    def update_discriminator(self) -> dict[str, float]:
        """
        如果开启 SIL 且判别器可用，则用当前 rollout 中的
        transition 条件输入 `(x_{t-1}, x_t)` 更新判别器。
        更新次数由自适应调度决定，且判别器恢复后自动停止本轮训练，
        避免策略一追上就再砸回去。
        """
        if not self.config.enable_sil:
            return {}
        if self.discriminator is None or self.discriminator_optimizer is None:
            return {}
        if self.rollout_buffer.imitation_obs is None or self.rollout_buffer.command_onehots is None:
            return {}

        buffer_ready, _, expert_transition_pool = self._get_discriminator_buffer_state()
        if not buffer_ready:
            return {
                "discriminator_buffer_ready": float(0.0),
                "discriminator_expert_transition_pool": float(expert_transition_pool),
                "discriminator_effective_batch_size": float(0.0),
            }

        policy_transition_inputs = self._collect_policy_transition_inputs_from_rollout()
        if policy_transition_inputs is None or policy_transition_inputs.numel() == 0:
            return {
                "discriminator_buffer_ready": float(0.0),
                "discriminator_expert_transition_pool": float(expert_transition_pool),
                "discriminator_effective_batch_size": float(0.0),
            }

        expert_score_sum = 0.0
        policy_score_sum = 0.0
        gp_sum = 0.0
        loss_sum = 0.0
        num_updates = 0

        batch_size = min(
            int(self.config.discriminator_batch_size),
            int(policy_transition_inputs.shape[0]),
            int(expert_transition_pool),
        )
        if batch_size <= 0:
            return {
                "discriminator_buffer_ready": float(0.0),
                "discriminator_expert_transition_pool": float(expert_transition_pool),
                "discriminator_effective_batch_size": float(0.0),
            }

        adaptive = bool(self.config.discriminator_adaptive_updates)
        target_updates = (
            self._compute_discriminator_update_count()
            if adaptive
            else int(self.config.discriminator_updates_per_iteration)
        )

        for _ in range(target_updates):
            (
                expert_prev_imitation,
                expert_curr_imitation,
                expert_prev_command_onehots,
                expert_curr_command_onehots,
            ) = self.sil_buffer.sample_transition_conditioned_torch(
                batch_size=batch_size,
                device=self.device,
            )
            expert_tensor = self._compose_discriminator_transition_torch_inputs(
                prev_imitation_obs=expert_prev_imitation,
                curr_imitation_obs=expert_curr_imitation,
                prev_command_onehot=expert_prev_command_onehots,
                curr_command_onehot=expert_curr_command_onehots,
            )

            sample_indices = torch.randint(
                low=0,
                high=policy_transition_inputs.shape[0],
                size=(batch_size,),
                device=self.device,
            )
            policy_tensor = policy_transition_inputs[sample_indices]

            loss, stats = self.discriminator.compute_loss(
                expert_samples=expert_tensor,
                policy_samples=policy_tensor,
            )

            self.discriminator_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.max_grad_norm)
            self.discriminator_optimizer.step()

            expert_score_sum += stats["expert_score"]
            policy_score_sum += stats["policy_score"]
            gp_sum += stats["gradient_penalty"]
            loss_sum += stats["total_loss"]
            num_updates += 1

            # 自适应模式下：如果本轮已经让 discriminator 恢复出明显区分度则提前退出
            if adaptive and num_updates >= 1:
                batch_separation = stats["expert_score"] - stats["policy_score"]
                if batch_separation > 0.40:
                    break

        avg_expert = expert_score_sum / max(num_updates, 1)
        avg_policy = policy_score_sum / max(num_updates, 1)
        self._last_disc_expert_score = avg_expert
        self._last_disc_policy_score = avg_policy
        self._last_disc_separation = avg_expert - avg_policy

        if num_updates == 0:
            return {
                "discriminator_buffer_ready": float(0.0),
                "discriminator_expert_transition_pool": float(expert_transition_pool),
                "discriminator_effective_batch_size": float(0.0),
            }

        return {
            "discriminator_buffer_ready": float(1.0),
            "discriminator_expert_transition_pool": float(expert_transition_pool),
            "discriminator_effective_batch_size": float(batch_size),
            "discriminator_loss": loss_sum / num_updates,
            "discriminator_expert_score": expert_score_sum / num_updates,
            "discriminator_policy_score": policy_score_sum / num_updates,
            "discriminator_score_margin": self._last_disc_separation,
            "discriminator_gradient_penalty": gp_sum / num_updates,
        }

    def _reset_env_with_new_command(self, seed: int | None = None) -> None:
        """
        为新 episode 采样 command 并 reset 环境。
        """
        self._current_command = self._sample_training_command()
        if self._use_tensor_env_api():
            observation, info = self.env.reset_tensor(
                command=self._current_command,
                seed=seed,
                lightweight_info=self.config.use_lightweight_tensor_info,
            )
            self._current_obs = self._ensure_batch_observation_tensor(observation)
            self._current_info = info
            self._previous_action = torch.zeros(
                (self.num_envs, self.env.action_dim),
                dtype=torch.float32,
                device=self.device,
            )
            return

        observation, info = self.env.reset(command=self._current_command, seed=seed)
        self._current_obs = self._ensure_batch_observation(observation)
        self._current_info = info
        if self.config.use_torch_reward_stack:
            self._previous_action = torch.zeros(
                (self.num_envs, self.env.action_dim),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self._previous_action = np.zeros((self.num_envs, self.env.action_dim), dtype=np.float32)

    def _sample_training_command(self) -> PasistCommand:
        """
        为下一段 rollout / episode 采样训练 command。

        逻辑：
        - 如果启用了 SkillSelector，则先由 selector 决定 skill 和速度
        - 否则直接走环境自己的 `sample_command()`
        """
        if self.skill_selector is None:
            return self.env.sample_command()

        sampled = self.skill_selector.sample_command()
        return self.env.build_command(
            velocity=sampled.velocity,
            skill_id=sampled.skill_id,
        )

    def _compute_step_reward_terms(
        self,
        action: np.ndarray,
        next_observation: np.ndarray,
        info: dict[str, Any],
    ) -> dict[str, float | np.ndarray]:
        """
        计算单步训练真正使用的奖励分解项。

        这是 PPO trainer 和 Isaac Lab 原生 env reward 的分界线：
        - 底层 env 仍会返回自己的 `env_reward`
        - 但 PPO 真正用来训练的是这里重新组合后的 total reward
        """
        active_command = self._current_command or info["command"]

        imitation_obs = self._ensure_batch_matrix(info["imitation_obs"], self.env.imitation_obs_dim)
        target_pose = np.asarray(info["target_pose"], dtype=np.float32)
        measured_velocity = info.get("measured_velocity", 0.0)
        base_height = info.get("base_height", None)
        base_pitch = info.get("base_pitch", None)
        current_critic_obs = self._ensure_batch_feature_array(info.get("critic_obs"), fallback=next_observation)
        previous_critic_obs = self._ensure_batch_feature_array(
            self._current_info.get("critic_obs") if self._current_info is not None else None,
            fallback=None,
        )

        base_angular_velocity = self._extract_base_angular_velocity(current_critic_obs)
        base_linear_velocity = self._extract_base_linear_velocity(current_critic_obs)
        projected_gravity = self._extract_projected_gravity(current_critic_obs)
        measured_yaw_rate = base_angular_velocity[:, 2]
        joint_velocity = self._extract_joint_velocity(current_critic_obs)
        previous_joint_velocity = (
            self._extract_joint_velocity(previous_critic_obs) if previous_critic_obs is not None else None
        )
        lateral_velocity = self._extract_lateral_velocity(current_critic_obs)

        target_base_height = self._extract_target_pose_base_height(target_pose)

        task_reward = compute_task_reward(
            imitation_observation=imitation_obs,
            target_pose=target_pose,
            commanded_velocity=active_command.velocity,
            measured_velocity=measured_velocity,
            measured_linear_velocity=base_linear_velocity,
            measured_angular_velocity=base_angular_velocity,
            projected_gravity=projected_gravity,
            skill_id=self._build_skill_id_vector(active_command.skill_id),
            command_skill_id=self._build_skill_id_vector(active_command.skill_id),
            pose_weight=self.config.pose_weight,
            velocity_weight=self.config.velocity_weight,
            command_weight=self.config.command_weight,
            pose_sigma=self.config.pose_sigma,
            velocity_sigma=self.config.velocity_sigma,
            commanded_yaw_rate=0.0,
            measured_yaw_rate=measured_yaw_rate,
            yaw_weight=self.config.yaw_weight,
            yaw_sigma=self.config.yaw_sigma,
            lin_vel_z_weight=self.config.lin_vel_z_weight,
            ang_vel_xy_weight=self.config.ang_vel_xy_weight,
            flat_orientation_weight=self.config.flat_orientation_weight,
            base_height=base_height,
            target_base_height=target_base_height,
            height_weight=self.config.height_weight,
            height_sigma=self.config.height_sigma,
        )

        regularization_reward = compute_regularization_reward(
            action=action,
            previous_action=self._previous_action,
            observation=next_observation,
            reference_observation=None,
            action_weight=self.config.action_weight,
            smoothness_weight=self.config.smoothness_weight,
            stability_weight=self.config.stability_weight,
            joint_velocity=joint_velocity,
            previous_joint_velocity=previous_joint_velocity,
            joint_acceleration_weight=self.config.joint_acceleration_weight,
            base_angular_velocity=base_angular_velocity,
            roll_pitch_rate_weight=self.config.roll_pitch_rate_weight,
            yaw_rate_weight=self.config.yaw_rate_weight,
            lateral_velocity=lateral_velocity,
            lateral_velocity_weight=self.config.lateral_velocity_weight,
        )

        command_onehot_batch = self._build_command_onehot_batch(active_command.one_hot)
        previous_imitation_obs = None
        if self._current_info is not None and "imitation_obs" in self._current_info:
            previous_imitation_obs = self._ensure_batch_matrix(
                self._current_info["imitation_obs"],
                self.env.imitation_obs_dim,
            )
        sil_reward = self._compute_sil_reward(
            previous_imitation_obs=previous_imitation_obs,
            current_imitation_obs=imitation_obs,
            previous_command_onehot=command_onehot_batch,
            current_command_onehot=command_onehot_batch,
        )
        omega_sil_terms = self._compute_omega_sil_terms()

        reward_terms = compute_reward_terms(
            task_reward=task_reward,
            sil_reward=sil_reward,
            regularization_reward=regularization_reward,
            sigma_t=self.config.sigma_t,
            omega_sil=omega_sil_terms["omega_sil"],
            omega_r=self.config.omega_r,
        )
        reward_terms["omega_sil_dtw"] = omega_sil_terms["omega_sil_dtw"]
        reward_terms["omega_sil_confidence"] = omega_sil_terms["omega_sil_confidence"]
        reward_terms["omega_sil_warmup"] = omega_sil_terms["omega_sil_warmup"]
        reward_terms["discriminator_score_margin"] = omega_sil_terms["discriminator_score_margin"]
        return reward_terms

    def _compute_step_reward_terms_torch(
        self,
        action: Any,
        next_observation: Any,
        info: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        active_command = self._current_command or info["command"]

        imitation_obs = self._ensure_batch_matrix_tensor(info["imitation_obs"], self.env.imitation_obs_dim)
        target_pose = self._ensure_target_pose_tensor(info["target_pose"])
        measured_velocity = info.get("measured_velocity", torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device))
        base_height = info.get("base_height", None)
        current_critic_obs = self._ensure_batch_feature_tensor(info.get("critic_obs"), fallback=next_observation)
        previous_critic_obs = self._ensure_batch_feature_tensor(
            self._current_info.get("critic_obs") if self._current_info is not None else None,
            fallback=None,
        )

        base_angular_velocity = self._extract_base_angular_velocity_tensor(current_critic_obs)
        base_linear_velocity = self._extract_base_linear_velocity_tensor(current_critic_obs)
        projected_gravity = self._extract_projected_gravity_tensor(current_critic_obs)
        measured_yaw_rate = base_angular_velocity[:, 2]
        joint_velocity = self._extract_joint_velocity_tensor(current_critic_obs)
        previous_joint_velocity = (
            self._extract_joint_velocity_tensor(previous_critic_obs) if previous_critic_obs is not None else None
        )
        lateral_velocity = self._extract_lateral_velocity_tensor(current_critic_obs)

        target_base_height = self._extract_target_pose_base_height_tensor(target_pose)

        task_reward = compute_task_reward_torch(
            imitation_observation=imitation_obs,
            target_pose=target_pose,
            commanded_velocity=active_command.velocity,
            measured_velocity=measured_velocity,
            measured_linear_velocity=base_linear_velocity,
            measured_angular_velocity=base_angular_velocity,
            projected_gravity=projected_gravity,
            skill_id=self._build_skill_id_vector_tensor(active_command.skill_id),
            command_skill_id=self._build_skill_id_vector_tensor(active_command.skill_id),
            pose_weight=self.config.pose_weight,
            velocity_weight=self.config.velocity_weight,
            command_weight=self.config.command_weight,
            pose_sigma=self.config.pose_sigma,
            velocity_sigma=self.config.velocity_sigma,
            commanded_yaw_rate=0.0,
            measured_yaw_rate=measured_yaw_rate,
            yaw_weight=self.config.yaw_weight,
            yaw_sigma=self.config.yaw_sigma,
            lin_vel_z_weight=self.config.lin_vel_z_weight,
            ang_vel_xy_weight=self.config.ang_vel_xy_weight,
            flat_orientation_weight=self.config.flat_orientation_weight,
            base_height=base_height,
            target_base_height=target_base_height,
            height_weight=self.config.height_weight,
            height_sigma=self.config.height_sigma,
        )

        regularization_reward = compute_regularization_reward_torch(
            action=action,
            previous_action=self._previous_action,
            observation=next_observation,
            reference_observation=None,
            action_weight=self.config.action_weight,
            smoothness_weight=self.config.smoothness_weight,
            stability_weight=self.config.stability_weight,
            joint_velocity=joint_velocity,
            previous_joint_velocity=previous_joint_velocity,
            joint_acceleration_weight=self.config.joint_acceleration_weight,
            base_angular_velocity=base_angular_velocity,
            roll_pitch_rate_weight=self.config.roll_pitch_rate_weight,
            yaw_rate_weight=self.config.yaw_rate_weight,
            lateral_velocity=lateral_velocity,
            lateral_velocity_weight=self.config.lateral_velocity_weight,
        )

        command_onehot_batch = self._build_command_onehot_batch_tensor(active_command.one_hot)
        previous_imitation_obs = None
        if self._current_info is not None and "imitation_obs" in self._current_info:
            previous_imitation_obs = self._ensure_batch_matrix_tensor(
                self._current_info["imitation_obs"],
                self.env.imitation_obs_dim,
            )
        sil_reward = self._compute_sil_reward_torch(
            previous_imitation_obs=previous_imitation_obs,
            current_imitation_obs=imitation_obs,
            previous_command_onehot=command_onehot_batch,
            current_command_onehot=command_onehot_batch,
        )
        omega_sil_terms = self._compute_omega_sil_terms()

        reward_terms = compute_reward_terms_torch(
            task_reward=task_reward,
            sil_reward=sil_reward,
            regularization_reward=regularization_reward,
            sigma_t=self.config.sigma_t,
            omega_sil=omega_sil_terms["omega_sil"],
            omega_r=self.config.omega_r,
        )
        reward_terms["omega_sil_dtw"] = torch.full_like(task_reward, float(omega_sil_terms["omega_sil_dtw"]))
        reward_terms["omega_sil_confidence"] = torch.full_like(
            task_reward,
            float(omega_sil_terms["omega_sil_confidence"]),
        )
        reward_terms["omega_sil_warmup"] = torch.full_like(task_reward, float(omega_sil_terms["omega_sil_warmup"]))
        reward_terms["discriminator_score_margin"] = torch.full_like(
            task_reward,
            float(omega_sil_terms["discriminator_score_margin"]),
        )
        return reward_terms

    def _compute_sil_reward(
        self,
        previous_imitation_obs: np.ndarray | None,
        current_imitation_obs: np.ndarray,
        previous_command_onehot: np.ndarray,
        current_command_onehot: np.ndarray,
    ) -> float | np.ndarray:
        """
        计算当前 step 的 SIL reward。

        如果判别器或 SIL buffer 还不可用，则安全地返回 0。
        当前 `r_SIL` 使用 transition 条件输入 `(x_{t-1}, x_t)`。
        """
        if not self.config.enable_sil:
            return np.zeros(self.num_envs, dtype=np.float32) if self.num_envs > 1 else 0.0
        if self.discriminator is None or len(self.sil_buffer) == 0:
            return np.zeros(self.num_envs, dtype=np.float32) if self.num_envs > 1 else 0.0
        if previous_imitation_obs is None:
            return np.zeros(self.num_envs, dtype=np.float32) if self.num_envs > 1 else 0.0
        buffer_ready, _, _ = self._get_discriminator_buffer_state()
        if not buffer_ready:
            return np.zeros(self.num_envs, dtype=np.float32) if self.num_envs > 1 else 0.0

        discriminator_inputs = self._compose_discriminator_transition_numpy_inputs(
            prev_imitation_obs=previous_imitation_obs,
            curr_imitation_obs=current_imitation_obs,
            prev_command_onehot=previous_command_onehot,
            curr_command_onehot=current_command_onehot,
        )
        imitation_tensor = torch.as_tensor(discriminator_inputs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            sil_reward = (
                self.discriminator.sil_reward(
                    imitation_tensor,
                    positive_margin=self.config.sil_reward_positive_margin,
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        if self.num_envs == 1 and sil_reward.shape[0] == 1:
            return float(sil_reward[0])
        return sil_reward

    def _compute_sil_reward_torch(
        self,
        previous_imitation_obs: torch.Tensor | None,
        current_imitation_obs: torch.Tensor,
        previous_command_onehot: torch.Tensor,
        current_command_onehot: torch.Tensor,
    ) -> torch.Tensor:
        if not self.config.enable_sil:
            return torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        if self.discriminator is None or len(self.sil_buffer) == 0:
            return torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        if previous_imitation_obs is None:
            return torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        buffer_ready, _, _ = self._get_discriminator_buffer_state()
        if not buffer_ready:
            return torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        discriminator_inputs = self._compose_discriminator_transition_torch_inputs(
            prev_imitation_obs=previous_imitation_obs,
            curr_imitation_obs=current_imitation_obs,
            prev_command_onehot=previous_command_onehot,
            curr_command_onehot=current_command_onehot,
        )
        with torch.no_grad():
            sil_reward = self.discriminator.sil_reward(
                discriminator_inputs,
                positive_margin=self.config.sil_reward_positive_margin,
            ).detach()
        return sil_reward.to(dtype=torch.float32, device=self.device)

    def _compute_omega_sil_terms(self) -> dict[str, float]:
        """
        组合当前 SIL reward 的全局门控项。

        当前使用三层收紧机制：
        - `omega_sil_dtw`: expert buffer DTW 质量门
        - `omega_sil_confidence`: 判别器 expert-policy 分离度门
        - `omega_sil_warmup`: expert buffer 数量 warmup 门
        """
        if not self.config.enable_sil:
            return {
                "omega_sil": 0.0,
                "omega_sil_dtw": 0.0,
                "omega_sil_confidence": 0.0,
                "omega_sil_warmup": 0.0,
                "discriminator_score_margin": 0.0,
            }
        buffer_ready, expert_trajectory_count, _ = self._get_discriminator_buffer_state()
        if not buffer_ready:
            return {
                "omega_sil": 0.0,
                "omega_sil_dtw": 0.0,
                "omega_sil_confidence": 0.0,
                "omega_sil_warmup": 0.0,
                "discriminator_score_margin": 0.0,
            }
        mean_dtw = compute_mean_sil_dtw(self.sil_buffer.summary())
        omega_sil_dtw = compute_sil_weight(
            mean_dtw_distance=mean_dtw,
            sigma_sil=self.config.sigma_sil,
            num_skills=self.env.num_skills,
            dtw_decay_rate=self.config.sil_weight_dtw_decay_rate,
        )
        discriminator_score_margin = float(self._last_disc_expert_score - self._last_disc_policy_score)
        omega_sil_confidence = compute_sil_confidence_weight(
            score_margin=discriminator_score_margin,
            min_margin=self.config.sil_weight_confidence_min_margin,
            max_margin=self.config.sil_weight_confidence_max_margin,
        )
        omega_sil_warmup = compute_sil_warmup_weight(
            expert_trajectory_count=expert_trajectory_count,
            warmup_start=self.config.min_sil_buffer_trajectories_for_discriminator,
            warmup_trajectories=self.config.sil_weight_warmup_trajectories,
        )
        omega_sil = omega_sil_dtw * omega_sil_confidence * omega_sil_warmup
        omega_sil = min(max(float(self.config.sil_weight_cap), 0.0), omega_sil)
        return {
            "omega_sil": float(omega_sil),
            "omega_sil_dtw": float(omega_sil_dtw),
            "omega_sil_confidence": float(omega_sil_confidence),
            "omega_sil_warmup": float(omega_sil_warmup),
            "discriminator_score_margin": float(discriminator_score_margin),
        }

    def _get_discriminator_buffer_state(self) -> tuple[bool, int, int]:
        """
        返回当前判别器 expert buffer 的可用状态。

        输出:
        - `buffer_ready`: 是否满足 warmup 后可用于 SIL reward / discriminator update
        - `expert_trajectory_count`: 当前 buffer 中 expert 轨迹数量
        - `expert_transition_pool`: 当前 buffer 中可采样的 expert transition 数
        """
        expert_trajectory_count = int(len(self.sil_buffer))
        expert_transition_pool = int(self.sil_buffer.num_transition_samples())
        warmup_threshold = int(max(self.config.min_sil_buffer_trajectories_for_discriminator, 0))
        buffer_ready = (
            expert_trajectory_count >= warmup_threshold
            and expert_transition_pool > 0
        )
        return bool(buffer_ready), expert_trajectory_count, expert_transition_pool

    def _resolve_device(self, device_name: str) -> torch.device:
        """
        将配置中的 device 字符串解析成 `torch.device`。
        """
        if str(device_name).startswith("cuda") and torch.cuda.is_available():
            return torch.device(device_name)
        return torch.device("cpu")

    def _normalize_feature_slices(
        self,
        feature_slices: list[tuple[int, int]] | list[list[int]] | tuple[tuple[int, int], ...] | None,
    ) -> list[tuple[int, int]] | None:
        """
        统一整理 DTW 特征切片配置。

        兼容来自 YAML 的常见写法：
        - `null`
        - `[[6, 18]]`
        - `[(6, 18)]`

        默认返回 `None`，表示保持当前 DTW 使用完整 imitation 特征的行为，
        从而不影响现有功能。
        """
        if feature_slices is None:
            return None

        normalized: list[tuple[int, int]] = []
        for item in feature_slices:
            if len(item) != 2:
                raise ValueError(
                    "trajectory_dtw_feature_slices 中的每个切片都必须有两个元素，例如 [6, 18]"
                )
            start, end = int(item[0]), int(item[1])
            normalized.append((start, end))
        return normalized

    def _use_tensor_env_api(self) -> bool:
        return bool(self.config.use_tensor_env_api) and hasattr(self.env, "reset_tensor") and hasattr(self.env, "step_tensor")

    def _to_tensor(self, value: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(device=self.device, dtype=dtype)
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _mean_scalar(self, value: Any) -> float:
        if torch.is_tensor(value):
            return float(value.detach().to(dtype=torch.float32).mean().item())
        return float(np.asarray(value, dtype=np.float32).mean())

    def _episode_is_tensor_dict(self, episode: dict[str, Any]) -> bool:
        for value in episode.values():
            if torch.is_tensor(value):
                return True
        return False

    def _extract_episode_skill_id(self, episode: dict[str, Any]) -> int:
        skill_values = episode.get("skill_ids")
        if skill_values is None:
            raise KeyError("episode 中缺少 skill_ids，无法提取 skill_id")
        if torch.is_tensor(skill_values):
            if skill_values.numel() == 0:
                raise ValueError("episode.skill_ids 为空，无法提取 skill_id")
            return int(skill_values.reshape(-1)[0].detach().to(dtype=torch.long).item())
        if not skill_values:
            raise ValueError("episode.skill_ids 为空，无法提取 skill_id")
        return int(self._to_numpy(skill_values[0], dtype=np.int64).reshape(-1)[0])

    def _episode_length(self, episode: dict[str, Any], key: str) -> int:
        value = episode.get(key)
        if value is None:
            return 0
        if torch.is_tensor(value):
            if value.ndim == 0:
                return int(value.numel())
            return int(value.shape[0])
        return int(len(value))

    def _episode_reward_tensor(self, episode: dict[str, Any], key: str) -> torch.Tensor:
        value = episode.get(key)
        if value is None:
            raise KeyError(f"episode 中缺少 {key}")
        if torch.is_tensor(value):
            return value.reshape(-1).detach().to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(
            [self._to_numpy(item, dtype=np.float32).reshape(-1)[0] for item in value],
            dtype=torch.float32,
            device=self.device,
        )

    def _ensure_target_pose_tensor(self, value: Any) -> torch.Tensor:
        pose = self._to_tensor(value, dtype=torch.float32)
        if pose.ndim == 1:
            return pose
        if pose.ndim == 2 and pose.shape[0] == self.num_envs:
            return pose
        raise ValueError(f"target_pose 维度不合法，实际 shape={tuple(pose.shape)}")

    def _ensure_batch_observation_tensor(self, observation: Any) -> torch.Tensor:
        return self._ensure_batch_matrix_tensor(observation, self.env.obs_dim)

    def _ensure_batch_critic_observation_tensor(self, value: Any, fallback: Any | None = None) -> torch.Tensor:
        critic_obs = self._ensure_batch_feature_tensor(value, fallback=fallback)
        if critic_obs is None:
            raise ValueError("critic observation 为空，无法更新 value function")
        if critic_obs.shape[1] != self.critic_obs_dim:
            raise ValueError(
                f"critic observation 特征维不匹配，期望 {self.critic_obs_dim}，实际得到 {critic_obs.shape[1]}"
            )
        return critic_obs.to(dtype=torch.float32)

    def _ensure_batch_matrix_tensor(self, value: Any, feature_dim: int) -> torch.Tensor:
        tensor = self._to_tensor(value, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.reshape(1, feature_dim)
        elif tensor.ndim != 2:
            raise ValueError(f"期望二维张量 [B, D]，实际得到 shape={tuple(tensor.shape)}")

        if tensor.shape[1] != feature_dim:
            raise ValueError(f"特征维不匹配，期望 {feature_dim}，实际得到 {tensor.shape[1]}")
        if tensor.shape[0] != self.num_envs:
            if tensor.shape[0] == 1 and self.num_envs > 1:
                tensor = tensor.expand(self.num_envs, -1)
            else:
                raise ValueError(f"batch 维不匹配，期望 {self.num_envs}，实际为 {tensor.shape[0]}")
        return tensor.to(dtype=torch.float32)

    def _ensure_batch_feature_tensor(self, value: Any, fallback: Any | None = None) -> torch.Tensor | None:
        source = fallback if value is None else value
        if source is None:
            return None

        tensor = self._to_tensor(source, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.reshape(1, -1)
        elif tensor.ndim != 2:
            raise ValueError(f"期望二维张量 [B, D]，实际得到 shape={tuple(tensor.shape)}")

        if tensor.shape[0] != self.num_envs:
            if tensor.shape[0] == 1 and self.num_envs > 1:
                tensor = tensor.expand(self.num_envs, -1)
            else:
                raise ValueError(f"batch 维不匹配，期望 {self.num_envs}，实际为 {tensor.shape[0]}")
        return tensor.to(dtype=torch.float32)

    def _extract_base_angular_velocity_tensor(self, critic_obs: torch.Tensor | None) -> torch.Tensor | None:
        if critic_obs is None or critic_obs.shape[1] < 6:
            return None
        return critic_obs[:, 3:6].to(dtype=torch.float32) / 0.2

    def _extract_base_linear_velocity_tensor(self, critic_obs: torch.Tensor | None) -> torch.Tensor | None:
        if critic_obs is None or critic_obs.shape[1] < 3:
            return None
        return critic_obs[:, 0:3].to(dtype=torch.float32)

    def _extract_joint_velocity_tensor(self, critic_obs: torch.Tensor | None) -> torch.Tensor | None:
        if critic_obs is None or critic_obs.shape[1] < 36:
            return None
        return critic_obs[:, 24:36].to(dtype=torch.float32) / 0.05

    def _extract_projected_gravity_tensor(self, critic_obs: torch.Tensor | None) -> torch.Tensor | None:
        if critic_obs is None or critic_obs.shape[1] < 9:
            return None
        return critic_obs[:, 6:9].to(dtype=torch.float32)

    def _extract_lateral_velocity_tensor(self, critic_obs: torch.Tensor | None) -> torch.Tensor | None:
        if critic_obs is None or critic_obs.shape[1] < 2:
            return None
        return critic_obs[:, 1].to(dtype=torch.float32)

    def _extract_target_pose_base_height_tensor(self, target_pose: Any) -> torch.Tensor | None:
        pose = self._to_tensor(target_pose, dtype=torch.float32)
        if pose.ndim == 0 or pose.numel() == 0:
            return None
        if pose.ndim == 1:
            return pose[0:1]
        return pose[:, 0].to(dtype=torch.float32)

    def _ensure_batch_column_tensor(self, value: Any) -> torch.Tensor:
        tensor = self._to_tensor(value, dtype=torch.float32)
        if tensor.ndim == 0:
            tensor = torch.full((self.num_envs, 1), float(tensor.item()), dtype=torch.float32, device=self.device)
        elif tensor.ndim == 1:
            if tensor.shape[0] == 1 and self.num_envs > 1:
                tensor = torch.full((self.num_envs, 1), float(tensor[0].item()), dtype=torch.float32, device=self.device)
            else:
                tensor = tensor.reshape(-1, 1)
        elif tensor.ndim == 2 and tensor.shape[1] == 1:
            pass
        else:
            raise ValueError(f"无法整理成列向量，实际 shape={tuple(tensor.shape)}")

        if tensor.shape[0] != self.num_envs:
            raise ValueError(f"batch 维不匹配，期望 {self.num_envs}，实际为 {tensor.shape[0]}")
        return tensor.to(dtype=torch.float32)

    def _ensure_batch_bool_tensor(self, value: Any) -> torch.Tensor:
        tensor = self._to_tensor(value, dtype=torch.bool)
        if tensor.ndim == 0:
            return torch.full((self.num_envs,), bool(tensor.item()), dtype=torch.bool, device=self.device)
        if tensor.ndim == 1:
            if tensor.shape[0] == 1 and self.num_envs > 1:
                return torch.full((self.num_envs,), bool(tensor[0].item()), dtype=torch.bool, device=self.device)
            return tensor
        if tensor.ndim == 2 and tensor.shape[1] == 1:
            return tensor[:, 0]
        raise ValueError(f"无法整理成 bool 向量，实际 shape={tuple(tensor.shape)}")

    def _build_skill_id_column_tensor(self, skill_id: int) -> torch.Tensor:
        return torch.full((self.num_envs, 1), int(skill_id), dtype=torch.long, device=self.device)

    def _build_skill_id_vector_tensor(self, skill_id: int) -> torch.Tensor:
        return torch.full((self.num_envs,), int(skill_id), dtype=torch.long, device=self.device)

    def _build_command_speed_column_tensor(self, velocity: float) -> torch.Tensor:
        return torch.full((self.num_envs, 1), float(velocity), dtype=torch.float32, device=self.device)

    def _build_command_onehot_batch_tensor(self, one_hot: np.ndarray) -> torch.Tensor:
        one_hot_tensor = self._to_tensor(one_hot, dtype=torch.float32).reshape(1, -1)
        return one_hot_tensor.expand(self.num_envs, -1)

    def _ensure_batch_observation(self, observation: Any) -> np.ndarray:
        """
        把 observation 统一整理成 shape = [num_envs, obs_dim]。
        """
        return self._ensure_batch_matrix(observation, self.env.obs_dim)

    def _ensure_batch_critic_observation(self, value: Any, fallback: Any | None = None) -> np.ndarray:
        """
        把 critic observation 统一整理成 shape = [num_envs, critic_obs_dim]。

        这里允许 critic 吃到比 actor 更多的特征，因此不能复用 policy observation 的维度检查。
        """
        critic_obs = self._ensure_batch_feature_array(value, fallback=fallback)
        if critic_obs is None:
            raise ValueError("critic observation 为空，无法更新 value function")
        if critic_obs.shape[1] != self.critic_obs_dim:
            raise ValueError(
                f"critic observation 特征维不匹配，期望 {self.critic_obs_dim}，实际得到 {critic_obs.shape[1]}"
            )
        return critic_obs.astype(np.float32, copy=False)

    def _ensure_batch_matrix(self, value: Any, feature_dim: int) -> np.ndarray:
        """
        将输入整理成二维数组 `[num_envs, feature_dim]`。
        """
        array = self._to_numpy(value, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, feature_dim)
        elif array.ndim != 2:
            raise ValueError(f"期望二维数组 [B, D]，实际得到 shape={array.shape}")

        if array.shape[1] != feature_dim:
            raise ValueError(f"特征维不匹配，期望 {feature_dim}，实际得到 {array.shape[1]}")
        return array.astype(np.float32, copy=False)

    def _ensure_batch_feature_array(self, value: Any, fallback: Any | None = None) -> np.ndarray | None:
        """
        将任意特征数组整理成 `[num_envs, D]`。

        这个辅助函数用于处理 `critic_obs` 这类“维度不一定等于 policy obs_dim”的输入，
        因此不能复用 `_ensure_batch_observation()`。
        """
        source = fallback if value is None else value
        if source is None:
            return None

        array = self._to_numpy(source, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        elif array.ndim != 2:
            raise ValueError(f"期望二维数组 [B, D]，实际得到 shape={array.shape}")

        if array.shape[0] != self.num_envs:
            if array.shape[0] == 1 and self.num_envs > 1:
                array = np.repeat(array, self.num_envs, axis=0)
            else:
                raise ValueError(f"batch 维不匹配，期望 {self.num_envs}，实际为 {array.shape[0]}")
        return array.astype(np.float32, copy=False)

    def _extract_base_angular_velocity(self, critic_obs: np.ndarray | None) -> np.ndarray | None:
        """
        从当前最小 critic observation 中提取未缩放的 `base_ang_vel`。

        约定:
        - `[3:6]` 对应 `base_ang_vel`
        - 当前观测缩放是 0.2，因此这里做一次反缩放
        """
        if critic_obs is None or critic_obs.shape[1] < 6:
            return None
        return critic_obs[:, 3:6].astype(np.float32, copy=False) / 0.2

    def _extract_base_linear_velocity(self, critic_obs: np.ndarray | None) -> np.ndarray | None:
        """
        从当前最小 critic observation 中提取 `base_lin_vel`。

        约定:
        - `[0:3]` 对应未缩放的 `base_lin_vel`
        """
        if critic_obs is None or critic_obs.shape[1] < 3:
            return None
        return critic_obs[:, 0:3].astype(np.float32, copy=False)

    def _extract_joint_velocity(self, critic_obs: np.ndarray | None) -> np.ndarray | None:
        """
        从当前最小 critic observation 中提取未缩放的 `joint_vel_rel`。

        约定:
        - `[24:36]` 对应 12 维关节速度
        - 当前观测缩放是 0.05，因此这里做一次反缩放
        """
        if critic_obs is None or critic_obs.shape[1] < 36:
            return None
        return critic_obs[:, 24:36].astype(np.float32, copy=False) / 0.05

    def _extract_projected_gravity(self, critic_obs: np.ndarray | None) -> np.ndarray | None:
        """
        从当前最小 critic observation 中提取 `projected_gravity`。

        约定:
        - `[6:9]` 对应 `projected_gravity`
        """
        if critic_obs is None or critic_obs.shape[1] < 9:
            return None
        return critic_obs[:, 6:9].astype(np.float32, copy=False)

    def _extract_lateral_velocity(self, critic_obs: np.ndarray | None) -> np.ndarray | None:
        """
        从 critic observation 中提取基座 y 向线速度。
        """
        if critic_obs is None or critic_obs.shape[1] < 2:
            return None
        return critic_obs[:, 1].astype(np.float32, copy=False)

    def _extract_target_pose_base_height(self, target_pose: Any) -> float | np.ndarray | None:
        """
        从 target pose 中提取目标 base height。

        当前 pose-only target pose 约定:
        - 第 0 维为 base_height
        - 第 1:13 维为 joint_pos_rel
        - 第 13 维为 skill_id
        """
        pose = self._to_numpy(target_pose, dtype=np.float32)
        if pose.ndim == 0 or pose.size == 0:
            return None
        if pose.ndim == 1:
            return float(pose[0])
        return pose[:, 0].astype(np.float32, copy=False)

    def _infer_discriminator_input_dim(self) -> int:
        """
        计算判别器的输入维度。

        当前约定:
        - imitation observation: [base_height, joint_pos_rel(12), skill_id]
        - 单时刻判别器帧: [base_height(1), joint_pos_rel(12), command_onehot(command_dim)]
        - 判别器真正使用 transition 输入: (x_{t-1}, x_t)
        """
        frame_dim = self._infer_discriminator_frame_dim()
        return frame_dim * 2

    def _infer_discriminator_frame_dim(self) -> int:
        """
        计算单时刻判别器帧的维度。
        """
        if int(self.env.imitation_obs_dim) < 13:
            raise ValueError(
                "当前 imitation_obs_dim 无法支持 joint_pos_rel 判别器输入；"
                f"需要至少 13 维，实际得到 {self.env.imitation_obs_dim}"
            )
        return 1 + 12 + int(self.env.command_dim)

    def _compose_discriminator_frame_numpy_inputs(
        self,
        imitation_obs: Any,
        command_onehot: Any,
    ) -> np.ndarray:
        """
        构造 numpy 版单时刻判别器输入。

        输入:
        - `imitation_obs`: [B, 14] = [base_height, joint_pos_rel(12), skill_id]
        - `command_onehot`: [B, command_dim] = 技能条件 one-hot

        输出:
        - [B, 1 + 12 + command_dim]
        """
        imitation_array = self._ensure_batch_matrix(imitation_obs, self.env.imitation_obs_dim)
        command_array = self._ensure_batch_matrix(command_onehot, self.env.command_dim)
        base_height = imitation_array[:, 0:1].astype(np.float32, copy=False)
        joint_pos_rel = imitation_array[:, 1:13].astype(np.float32, copy=False)
        joint_pos_rel = (joint_pos_rel / float(self._discriminator_joint_pos_scale)).astype(np.float32, copy=False)
        return np.concatenate([base_height, joint_pos_rel, command_array], axis=-1).astype(np.float32, copy=False)

    def _compose_discriminator_transition_numpy_inputs(
        self,
        prev_imitation_obs: Any,
        curr_imitation_obs: Any,
        prev_command_onehot: Any,
        curr_command_onehot: Any,
    ) -> np.ndarray:
        """
        构造 numpy 版 transition 判别器输入。

        输出维度:
        - [B, 2 * (1 + 12 + command_dim)]
        """
        previous_frame = self._compose_discriminator_frame_numpy_inputs(
            imitation_obs=prev_imitation_obs,
            command_onehot=prev_command_onehot,
        )
        current_frame = self._compose_discriminator_frame_numpy_inputs(
            imitation_obs=curr_imitation_obs,
            command_onehot=curr_command_onehot,
        )
        return np.concatenate([previous_frame, current_frame], axis=-1).astype(np.float32, copy=False)

    def _compose_discriminator_frame_torch_inputs(
        self,
        imitation_obs: torch.Tensor,
        command_onehot: torch.Tensor,
    ) -> torch.Tensor:
        """
        构造 torch 版单时刻判别器输入。

        输出维度:
        - [B, 1 + 12 + command_dim]
        """
        if imitation_obs.ndim != 2:
            raise ValueError(f"判别器 imitation_obs 期望二维张量 [B, D]，实际得到 shape={tuple(imitation_obs.shape)}")
        if imitation_obs.shape[1] != self.env.imitation_obs_dim:
            raise ValueError(
                f"判别器 imitation_obs 特征维不匹配，期望 {self.env.imitation_obs_dim}，"
                f"实际得到 {imitation_obs.shape[1]}"
            )
        if command_onehot.ndim != 2:
            raise ValueError(
                f"判别器 command_onehot 期望二维张量 [B, D]，实际得到 shape={tuple(command_onehot.shape)}"
            )
        if command_onehot.shape[1] != self.env.command_dim:
            raise ValueError(
                f"判别器 command_onehot 特征维不匹配，期望 {self.env.command_dim}，"
                f"实际得到 {command_onehot.shape[1]}"
            )
        base_height = imitation_obs[:, 0:1]
        joint_pos_rel = imitation_obs[:, 1:13] / float(self._discriminator_joint_pos_scale)
        return torch.cat([base_height, joint_pos_rel, command_onehot], dim=-1)

    def _compose_discriminator_transition_torch_inputs(
        self,
        prev_imitation_obs: torch.Tensor,
        curr_imitation_obs: torch.Tensor,
        prev_command_onehot: torch.Tensor,
        curr_command_onehot: torch.Tensor,
    ) -> torch.Tensor:
        """
        构造 torch 版 transition 判别器输入。

        输出维度:
        - [B, 2 * (1 + 12 + command_dim)]
        """
        previous_frame = self._compose_discriminator_frame_torch_inputs(
            imitation_obs=prev_imitation_obs,
            command_onehot=prev_command_onehot,
        )
        current_frame = self._compose_discriminator_frame_torch_inputs(
            imitation_obs=curr_imitation_obs,
            command_onehot=curr_command_onehot,
        )
        return torch.cat([previous_frame, current_frame], dim=-1)

    def _collect_policy_transition_inputs_from_rollout(self) -> torch.Tensor | None:
        """
        从当前 rollout 中提取可用于 transition 判别器训练的 policy 样本。
        """
        episodes = self.rollout_buffer.extract_episodes_tensor()
        transition_inputs: list[torch.Tensor] = []
        for episode in episodes:
            if "imitation_obs" not in episode:
                continue
            if self._episode_length(episode, "imitation_obs") < 2:
                continue

            if self._episode_is_tensor_dict(episode):
                imitation_sequence = self._to_tensor(episode["imitation_obs"], dtype=torch.float32)
                if imitation_sequence.ndim == 1:
                    imitation_sequence = imitation_sequence.reshape(1, -1)
                elif imitation_sequence.ndim != 2:
                    raise ValueError(
                        f"episode imitation_obs 维度不正确，期望 [T, D]，实际得到 shape={tuple(imitation_sequence.shape)}"
                    )
                command_sequence = self._extract_episode_command_onehot_sequence_tensor(episode)
                transition_inputs.append(
                    self._compose_discriminator_transition_torch_inputs(
                        prev_imitation_obs=imitation_sequence[:-1],
                        curr_imitation_obs=imitation_sequence[1:],
                        prev_command_onehot=command_sequence[:-1],
                        curr_command_onehot=command_sequence[1:],
                    )
                )
                continue

            imitation_sequence = np.asarray(
                [self._to_numpy(item, dtype=np.float32).reshape(-1) for item in episode["imitation_obs"]],
                dtype=np.float32,
            )
            command_sequence = self._extract_episode_command_onehot_sequence(episode)
            transition_inputs.append(
                torch.as_tensor(
                    self._compose_discriminator_transition_numpy_inputs(
                        prev_imitation_obs=imitation_sequence[:-1],
                        curr_imitation_obs=imitation_sequence[1:],
                        prev_command_onehot=command_sequence[:-1],
                        curr_command_onehot=command_sequence[1:],
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
            )

        if not transition_inputs:
            return None

        return torch.cat(transition_inputs, dim=0)

    def _extract_episode_command_onehot_sequence(self, episode: dict[str, Any]) -> np.ndarray:
        """
        从 rollout 提取的 episode 中恢复 `[T, command_dim]` 的技能 one-hot 序列。

        优先使用 rollout buffer 中显式保存的 `command_onehots`。
        如果旧数据或特殊路径没有保存该字段，则退回到 `skill_ids -> one-hot` 的确定性映射，
        保证条件判别器的 policy 分支仍然可以构造输入。
        """
        if "command_onehots" in episode and episode["command_onehots"]:
            command_sequence = np.asarray(
                [self._to_numpy(item, dtype=np.float32).reshape(-1) for item in episode["command_onehots"]],
                dtype=np.float32,
            )
        elif "skill_ids" in episode and episode["skill_ids"]:
            skill_id_sequence = np.asarray(
                [int(self._to_numpy(item, dtype=np.int64).reshape(-1)[0]) for item in episode["skill_ids"]],
                dtype=np.int64,
            )
            command_sequence = np.asarray(
                [self.env.get_skill_one_hot(int(skill_id)).reshape(-1) for skill_id in skill_id_sequence],
                dtype=np.float32,
            )
        else:
            raise KeyError("episode 中既没有 command_onehots，也没有 skill_ids，无法构造条件判别器输入")

        if command_sequence.ndim != 2 or command_sequence.shape[1] != self.env.command_dim:
            raise ValueError(
                "episode command_onehot 序列维度不正确，"
                f"期望 [T, {self.env.command_dim}]，实际得到 shape={command_sequence.shape}"
            )
        return command_sequence.astype(np.float32, copy=False)

    def _extract_episode_command_onehot_sequence_tensor(self, episode: dict[str, Any]) -> torch.Tensor:
        """
        从 tensor 版 episode 中恢复 `[T, command_dim]` 的技能 one-hot 序列。
        """
        if "command_onehots" in episode and episode["command_onehots"] is not None:
            command_sequence = self._to_tensor(episode["command_onehots"], dtype=torch.float32)
            if command_sequence.ndim == 1:
                command_sequence = command_sequence.reshape(1, -1)
            elif command_sequence.ndim != 2:
                raise ValueError(
                    "episode command_onehot 序列维度不正确，"
                    f"期望 [T, {self.env.command_dim}]，实际得到 shape={tuple(command_sequence.shape)}"
                )
        elif "skill_ids" in episode and episode["skill_ids"] is not None:
            skill_id_sequence = self._to_tensor(episode["skill_ids"], dtype=torch.long).reshape(-1)
            command_vectors = [
                self._to_tensor(self.env.get_skill_one_hot(int(skill_id.item())), dtype=torch.float32)
                for skill_id in skill_id_sequence
            ]
            command_sequence = torch.stack(command_vectors, dim=0)
        else:
            raise KeyError("episode 中既没有 command_onehots，也没有 skill_ids，无法构造条件判别器输入")

        if command_sequence.ndim != 2 or command_sequence.shape[1] != self.env.command_dim:
            raise ValueError(
                "episode command_onehot 序列维度不正确，"
                f"期望 [T, {self.env.command_dim}]，实际得到 shape={tuple(command_sequence.shape)}"
            )
        return command_sequence.to(dtype=torch.float32, device=self.device)

    def _ensure_batch_column(self, value: Any) -> np.ndarray:
        """
        将标量或向量整理成 `[num_envs, 1]`。
        """
        array = self._to_numpy(value, dtype=np.float32)
        if array.ndim == 0:
            array = np.full((self.num_envs, 1), float(array), dtype=np.float32)
        elif array.ndim == 1:
            if array.shape[0] == 1 and self.num_envs > 1:
                array = np.full((self.num_envs, 1), float(array[0]), dtype=np.float32)
            else:
                array = array.reshape(-1, 1)
        elif array.ndim == 2 and array.shape[1] == 1:
            pass
        else:
            raise ValueError(f"无法整理成列向量，实际 shape={array.shape}")

        if array.shape[0] != self.num_envs:
            raise ValueError(f"batch 维不匹配，期望 {self.num_envs}，实际为 {array.shape[0]}")
        return array.astype(np.float32, copy=False)

    def _ensure_batch_bool(self, value: Any) -> np.ndarray:
        """
        将 bool 标量或向量整理成 shape = [num_envs] 的布尔数组。
        """
        array = np.asarray(value, dtype=bool)
        if array.ndim == 0:
            return np.full((self.num_envs,), bool(array), dtype=bool)
        if array.ndim == 1:
            if array.shape[0] == 1 and self.num_envs > 1:
                return np.full((self.num_envs,), bool(array[0]), dtype=bool)
            return array
        if array.ndim == 2 and array.shape[1] == 1:
            return array[:, 0]
        raise ValueError(f"无法整理成 bool 向量，实际 shape={array.shape}")

    def _build_skill_id_column(self, skill_id: int) -> np.ndarray:
        """
        构造 `[num_envs, 1]` 的 skill_id 列向量。
        """
        return np.full((self.num_envs, 1), int(skill_id), dtype=np.int64)

    def _build_skill_id_vector(self, skill_id: int) -> np.ndarray:
        """
        构造 `[num_envs]` 的 skill_id 向量。
        """
        return np.full((self.num_envs,), int(skill_id), dtype=np.int64)

    def _build_command_speed_column(self, velocity: float) -> np.ndarray:
        """
        构造 `[num_envs, 1]` 的速度命令列向量。
        """
        return np.full((self.num_envs, 1), float(velocity), dtype=np.float32)

    def _build_command_onehot_batch(self, one_hot: np.ndarray) -> np.ndarray:
        """
        构造 `[num_envs, command_dim]` 的 one-hot batch。
        """
        one_hot_array = np.asarray(one_hot, dtype=np.float32).reshape(1, -1)
        return np.repeat(one_hot_array, self.num_envs, axis=0)

    def _to_numpy(self, value: Any, dtype=np.float32) -> np.ndarray:
        """
        将 torch / numpy / Python 标量统一转成 numpy。
        """
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return np.asarray(value, dtype=dtype)
