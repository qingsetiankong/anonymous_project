from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
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
    compute_sil_weight,
    compute_task_reward,
)
from rl.actor_critic_new import MLP
from rl.rollout_buffer import RolloutBuffer
from sil.discriminator import SILDiscriminator
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
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.0
    device: str = "cpu"

    # policy / value 网络结构
    actor_hidden_dims: tuple[int, ...] = (256, 256)
    critic_hidden_dims: tuple[int, ...] = (256, 256)
    init_log_std: float = -0.5

    # task reward 相关超参数
    pose_weight: float = 1.0
    velocity_weight: float = 0.0
    command_weight: float = 0.0
    pose_sigma: float = 1.0
    velocity_sigma: float = 0.25

    # regularization reward 相关超参数
    action_weight: float = 1.0e-3
    smoothness_weight: float = 1.0e-3
    stability_weight: float = 0.0

    # total reward 动态权重相关超参数
    sigma_t: float = 0.5
    sigma_sil: float = 0.5
    omega_r: float = 1.0

    # PASIST / SIL 可选开关
    enable_sil: bool = True
    use_skill_selector: bool = False
    discriminator_config_path: str = "configs/training/discriminator.yaml"
    discriminator_batch_size: int = 256
    discriminator_updates_per_iteration: int = 1
    sil_buffer_capacity_per_skill: int = 8
    trajectory_dtw_weight: float = 0.1
    trajectory_reference_length_ratio: float = 0.5
    trajectory_normalize_dtw: bool = True
    trajectory_dtw_feature_slices: list[tuple[int, int]] | None = None


class GaussianPolicy(nn.Module):
    """
    连续动作 PPO 使用的对角高斯策略网络。

    输入:
    - observation，shape = [batch_size, obs_dim]

    输出:
    - action mean，shape = [batch_size, action_dim]
    - log_std 是可学习的全局参数，shape = [action_dim]
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims, init_log_std: float = -0.5) -> None:
        super().__init__()
        self.mean_net = MLP(
            input_dim=obs_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
            output_activation=None,
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

    def __init__(self, obs_dim: int, hidden_dims) -> None:
        super().__init__()
        self.value_net = MLP(
            input_dim=obs_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            output_activation=None,
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

        self.policy = GaussianPolicy(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            hidden_dims=self.config.actor_hidden_dims,
            init_log_std=self.config.init_log_std,
        ).to(self.device)
        self.value_function = ValueFunction(
            obs_dim=env.obs_dim,
            hidden_dims=self.config.critic_hidden_dims,
        ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.value_function.parameters(), lr=self.config.critic_lr)

        self.rollout_buffer = RolloutBuffer(
            num_steps=self.config.num_steps,
            num_envs=self.num_envs,
            obs_dim=env.obs_dim,
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

        self.sil_buffer = sil_buffer or SILBuffer(
            capacity_per_skill=self.config.sil_buffer_capacity_per_skill,
        )
        self.trajectory_selector = trajectory_selector or TrajectorySelector(
            dtw_weight=self.config.trajectory_dtw_weight,
            reference_length_ratio=self.config.trajectory_reference_length_ratio,
            normalize_dtw=self.config.trajectory_normalize_dtw,
            dtw_feature_slices=self._normalize_feature_slices(self.config.trajectory_dtw_feature_slices),
        )

        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        if self.config.enable_sil and self.discriminator is None:
            config_path = Path(self.config.discriminator_config_path)
            if config_path.exists():
                self.discriminator = SILDiscriminator.from_yaml(
                    input_dim=env.imitation_obs_dim,
                    config_path=config_path,
                ).to(self.device)
                self.discriminator_optimizer = self.discriminator.build_optimizer(config_path=config_path)

        self._current_obs: np.ndarray | None = None
        self._current_info: dict[str, Any] | None = None
        self._current_command: PasistCommand | None = None
        self._previous_action = np.zeros((self.num_envs, env.action_dim), dtype=np.float32)
        self._episode_ids = np.zeros(self.num_envs, dtype=np.int64)

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
            stats = self.train_iteration(seed=seed if iteration == 0 else None)
            stats["iteration"] = float(iteration)
            history.append(stats)
            if on_iteration_end is not None:
                on_iteration_end(dict(stats))
        return history

    def train_iteration(self, seed: int | None = None) -> dict[str, float]:
        """
        执行一次完整训练迭代。

        迭代内容：
        1. 收集 rollout
        2. 进行 PPO 更新
        3. 用 rollout 轨迹更新 SIL buffer
        4. 可选：更新 discriminator
        5. 清空 rollout buffer
        """
        if self._current_obs is None:
            self._reset_env_with_new_command(seed=seed)

        rollout_stats = self.collect_rollout()
        ppo_stats = self.update_policy()
        sil_stats = self.update_sil_components()

        self.rollout_buffer.clear()

        merged = {}
        merged.update(rollout_stats)
        merged.update(ppo_stats)
        merged.update(sil_stats)
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
        done_count = 0.0

        for _ in range(self.config.num_steps):
            obs_batch = self._ensure_batch_observation(self._current_obs)
            obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                actions_tensor, log_probs_tensor = self.policy.sample(obs_tensor)
                values_tensor = self.value_function(obs_tensor)

            actions_np = actions_tensor.detach().cpu().numpy().astype(np.float32)
            next_obs, _, terminated, truncated, info = self.env.step(actions_np if self.num_envs > 1 else actions_np[0])

            next_obs_batch = self._ensure_batch_observation(next_obs)
            reward_terms = self._compute_step_reward_terms(
                action=actions_np,
                next_observation=next_obs_batch,
                info=info,
            )

            done_array = np.logical_or(self._ensure_batch_bool(terminated), self._ensure_batch_bool(truncated))
            reward_total = self._ensure_batch_column(reward_terms["total_reward"])
            reward_task = self._ensure_batch_column(reward_terms["task_reward"])
            reward_reg = self._ensure_batch_column(reward_terms["regularization_reward"])
            reward_sil = self._ensure_batch_column(reward_terms["sil_reward"])

            active_command = self._current_command or self._sample_training_command()
            self.rollout_buffer.add(
                obs=torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device),
                action=torch.as_tensor(actions_np, dtype=torch.float32, device=self.device),
                log_prob=log_probs_tensor.detach(),
                value=values_tensor.detach(),
                reward_total=torch.as_tensor(reward_total, dtype=torch.float32, device=self.device),
                reward_task=torch.as_tensor(reward_task, dtype=torch.float32, device=self.device),
                reward_reg=torch.as_tensor(reward_reg, dtype=torch.float32, device=self.device),
                reward_sil=torch.as_tensor(reward_sil, dtype=torch.float32, device=self.device),
                done=torch.as_tensor(self._ensure_batch_column(done_array), dtype=torch.float32, device=self.device),
                next_obs=torch.as_tensor(next_obs_batch, dtype=torch.float32, device=self.device),
                skill_id=torch.as_tensor(self._build_skill_id_column(active_command.skill_id), dtype=torch.long, device=self.device),
                imitation_obs=torch.as_tensor(
                    self._ensure_batch_matrix(info["imitation_obs"], self.env.imitation_obs_dim),
                    dtype=torch.float32,
                    device=self.device,
                ),
                episode_id=torch.as_tensor(self._episode_ids.reshape(self.num_envs, 1), dtype=torch.long, device=self.device),
                terminated=torch.as_tensor(
                    self._ensure_batch_column(self._ensure_batch_bool(terminated)),
                    dtype=torch.float32,
                    device=self.device,
                ),
                truncated=torch.as_tensor(
                    self._ensure_batch_column(self._ensure_batch_bool(truncated)),
                    dtype=torch.float32,
                    device=self.device,
                ),
                command_speed=torch.as_tensor(
                    self._build_command_speed_column(active_command.velocity),
                    dtype=torch.float32,
                    device=self.device,
                ),
                command_onehot=torch.as_tensor(
                    self._build_command_onehot_batch(active_command.one_hot),
                    dtype=torch.float32,
                    device=self.device,
                ),
            )

            reward_total_sum += float(np.asarray(reward_terms["total_reward"]).mean())
            reward_task_sum += float(np.asarray(reward_terms["task_reward"]).mean())
            reward_reg_sum += float(np.asarray(reward_terms["regularization_reward"]).mean())
            reward_sil_sum += float(np.asarray(reward_terms["sil_reward"]).mean())
            omega_t_sum += float(np.asarray(reward_terms["omega_t"]).mean())
            omega_sil_sum += float(np.asarray(reward_terms["omega_sil"]).mean())
            done_count += float(done_array.sum())

            self._current_obs = next_obs_batch
            self._current_info = info
            self._previous_action = actions_np.copy()

            # 当前版本最稳的 episode 切换逻辑：
            # - 单环境时，done 后显式 reset，并为下一个 episode 采样新 command
            # - 多环境时，底层 Isaac Lab 已经自动 reset，对单技能版本保持当前 command 不变
            if np.any(done_array):
                done_indices = np.nonzero(done_array)[0]
                self._episode_ids[done_indices] += 1

                if self.num_envs == 1:
                    self._reset_env_with_new_command(seed=None)
                else:
                    self._previous_action[done_indices] = 0.0

        return {
            "rollout_reward_total_mean": reward_total_sum / max(self.config.num_steps, 1),
            "rollout_reward_task_mean": reward_task_sum / max(self.config.num_steps, 1),
            "rollout_reward_reg_mean": reward_reg_sum / max(self.config.num_steps, 1),
            "rollout_reward_sil_mean": reward_sil_sum / max(self.config.num_steps, 1),
            "rollout_omega_t_mean": omega_t_sum / max(self.config.num_steps, 1),
            "rollout_omega_sil_mean": omega_sil_sum / max(self.config.num_steps, 1),
            "rollout_done_count": done_count,
        }

    def update_policy(self) -> dict[str, float]:
        """
        使用当前 rollout buffer 做 PPO 更新。
        """
        if self._current_obs is None:
            return {}

        last_obs_tensor = torch.as_tensor(
            self._ensure_batch_observation(self._current_obs),
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

                values = self.value_function(observations)
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

    def update_sil_components(self) -> dict[str, float]:
        """
        使用本轮 rollout 更新 PASIST 的轨迹筛选和判别器模块。

        当前流程：
        1. 从 rollout buffer 中提 episode
        2. 用 trajectory selector 计算 DTW 和 assessment score
        3. 接纳高质量轨迹到 SIL buffer
        4. 如果开启了 discriminator，则再做若干次更新
        """
        episodes = self.rollout_buffer.extract_episodes()
        accepted_count = 0
        evaluated_count = 0

        for episode in episodes:
            trajectory = self.trajectory_selector.episode_from_buffer_dict(episode)
            target_pose = self.env.get_target_pose(trajectory.skill_id)
            evaluation = self.trajectory_selector.evaluate(trajectory=trajectory, target_pose=target_pose)
            evaluated_count += 1

            if self.skill_selector is not None:
                self.skill_selector.update(skill_id=trajectory.skill_id, task_reward=evaluation.task_return)

            if evaluation.accepted:
                accepted_count += 1
                self.sil_buffer.add_episode(
                    episode=episode,
                    assessment_score=evaluation.assessment_score,
                    skill_id=trajectory.skill_id,
                    metadata={
                        "dtw_distance": evaluation.dtw_distance,
                        "task_return": evaluation.task_return,
                        "trajectory_length": evaluation.trajectory_length,
                    },
                )

        discriminator_stats = self.update_discriminator()
        summary = self.sil_buffer.summary()

        stats = {
            "trajectory_evaluated_count": float(evaluated_count),
            "trajectory_accepted_count": float(accepted_count),
            "sil_buffer_num_trajectories": float(len(self.sil_buffer)),
            "sil_buffer_mean_dtw": float(compute_mean_sil_dtw(summary)),
        }
        stats.update(discriminator_stats)
        return stats

    def update_discriminator(self) -> dict[str, float]:
        """
        如果开启 SIL 且判别器可用，则用当前 rollout 中的 imitation_obs 更新判别器。
        """
        if not self.config.enable_sil:
            return {}
        if self.discriminator is None or self.discriminator_optimizer is None:
            return {}
        if len(self.sil_buffer) == 0 or self.rollout_buffer.imitation_obs is None:
            return {}

        valid_policy_samples = self.rollout_buffer._flatten_valid(self.rollout_buffer.imitation_obs)
        if valid_policy_samples.numel() == 0:
            return {}

        expert_score_sum = 0.0
        policy_score_sum = 0.0
        gp_sum = 0.0
        loss_sum = 0.0
        num_updates = 0

        batch_size = min(self.config.discriminator_batch_size, valid_policy_samples.shape[0])
        for _ in range(self.config.discriminator_updates_per_iteration):
            expert_samples = self.sil_buffer.sample(batch_size=batch_size)
            expert_tensor = torch.as_tensor(expert_samples, dtype=torch.float32, device=self.device)

            sample_indices = torch.randint(
                low=0,
                high=valid_policy_samples.shape[0],
                size=(batch_size,),
                device=self.device,
            )
            policy_tensor = valid_policy_samples[sample_indices]

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

        if num_updates == 0:
            return {}

        return {
            "discriminator_loss": loss_sum / num_updates,
            "discriminator_expert_score": expert_score_sum / num_updates,
            "discriminator_policy_score": policy_score_sum / num_updates,
            "discriminator_gradient_penalty": gp_sum / num_updates,
        }

    def _reset_env_with_new_command(self, seed: int | None = None) -> None:
        """
        为新 episode 采样 command 并 reset 环境。
        """
        self._current_command = self._sample_training_command()
        observation, info = self.env.reset(command=self._current_command, seed=seed)
        self._current_obs = self._ensure_batch_observation(observation)
        self._current_info = info
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

        task_reward = compute_task_reward(
            imitation_observation=imitation_obs,
            target_pose=target_pose,
            commanded_velocity=active_command.velocity,
            measured_velocity=measured_velocity,
            skill_id=self._build_skill_id_vector(active_command.skill_id),
            command_skill_id=self._build_skill_id_vector(active_command.skill_id),
            pose_weight=self.config.pose_weight,
            velocity_weight=self.config.velocity_weight,
            command_weight=self.config.command_weight,
            pose_sigma=self.config.pose_sigma,
            velocity_sigma=self.config.velocity_sigma,
        )

        regularization_reward = compute_regularization_reward(
            action=action,
            previous_action=self._previous_action,
            observation=next_observation,
            reference_observation=None,
            action_weight=self.config.action_weight,
            smoothness_weight=self.config.smoothness_weight,
            stability_weight=self.config.stability_weight,
        )

        sil_reward = self._compute_sil_reward(imitation_obs)
        omega_sil = self._compute_omega_sil()

        return compute_reward_terms(
            task_reward=task_reward,
            sil_reward=sil_reward,
            regularization_reward=regularization_reward,
            sigma_t=self.config.sigma_t,
            omega_sil=omega_sil,
            omega_r=self.config.omega_r,
        )

    def _compute_sil_reward(self, imitation_obs: np.ndarray) -> float | np.ndarray:
        """
        计算当前 step 的 SIL reward。

        如果判别器或 SIL buffer 还不可用，则安全地返回 0。
        """
        if not self.config.enable_sil:
            return np.zeros(self.num_envs, dtype=np.float32) if self.num_envs > 1 else 0.0
        if self.discriminator is None or len(self.sil_buffer) == 0:
            return np.zeros(self.num_envs, dtype=np.float32) if self.num_envs > 1 else 0.0

        imitation_tensor = torch.as_tensor(imitation_obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            sil_reward = self.discriminator.sil_reward(imitation_tensor).detach().cpu().numpy().astype(np.float32)
        if self.num_envs == 1 and sil_reward.shape[0] == 1:
            return float(sil_reward[0])
        return sil_reward

    def _compute_omega_sil(self) -> float:
        """
        根据当前 SIL buffer 的 DTW 统计计算 `omega_sil`。
        """
        if not self.config.enable_sil:
            return 0.0
        mean_dtw = compute_mean_sil_dtw(self.sil_buffer.summary())
        return compute_sil_weight(
            mean_dtw_distance=mean_dtw,
            sigma_sil=self.config.sigma_sil,
            num_skills=self.env.num_skills,
        )

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

    def _ensure_batch_observation(self, observation: Any) -> np.ndarray:
        """
        把 observation 统一整理成 shape = [num_envs, obs_dim]。
        """
        return self._ensure_batch_matrix(observation, self.env.obs_dim)

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
