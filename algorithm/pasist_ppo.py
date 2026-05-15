from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rl.pasist_actor_critic import PasistActorCritic
from runner.rollout_storage import RolloutStorage


class PasistPPO:
    actor_critic: PasistActorCritic

    def __init__(
        self,
        actor_critic: PasistActorCritic,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        mini_batch_size: int = 256,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        device: str = "cpu",
    ):
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        self.transition = RolloutStorage.Transition()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.mini_batch_size = mini_batch_size
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: list[int],
        privileged_obs_shape: list[int],
        action_shape: list[int],
    ):
        self.storage = RolloutStorage(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            actor_obs_shape=actor_obs_shape,
            privileged_obs_shape=privileged_obs_shape,
            action_shape=action_shape,
            device=self.device,
        )

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = (
            self.actor_critic.get_actions_log_prob(
                self.transition.actions
            ).detach()
        )
        self.transition.action_mean = self.actor_critic.actor.action_mean
        if self.transition.action_mean is None:
            self.transition.action_mean = torch.zeros_like(self.transition.actions)
        self.transition.action_std = self.actor_critic.actor.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(
        self,
        next_obs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict,
    ):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones.clone()
        if "time_outs" in infos:
            time_outs = infos["time_outs"].to(self.device)
            self.transition.rewards += (
                self.gamma * self.transition.values * time_outs.unsqueeze(1)
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def compute_returns(self, last_critic_obs: torch.Tensor):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self) -> tuple[float, float, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0

        generator = self.storage.mini_batch_generator(
            self.mini_batch_size, self.num_learning_epochs
        )
        num_updates = 0
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            _target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            actions_log_prob_batch, entropy_batch = (
                self.actor_critic.actor.log_prob_and_entropy(
                    obs_batch, actions_batch
                )
            )
            value_batch = self.actor_critic.evaluate(critic_obs_batch)

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(self.actor_critic.actor.action_std / old_sigma_batch + 1e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - self.actor_critic.actor.action_mean)
                        )
                        / (2.0 * torch.square(self.actor_critic.actor.action_std))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            surrogate_loss = self._compute_surrogate_loss(
                actions_log_prob_batch, old_actions_log_prob_batch, advantages_batch
            )
            value_loss = self._compute_value_loss(
                value_batch, returns_batch
            )
            entropy_loss = -self.entropy_coef * entropy_batch.mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                + entropy_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), self.max_grad_norm
            )
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            num_updates += 1

        if num_updates == 0:
            return 0.0, 0.0, 0.0
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates

        self.storage.clear()
        return mean_value_loss, mean_surrogate_loss, mean_entropy

    def _compute_surrogate_loss(
        self,
        actions_log_prob_batch: torch.Tensor,
        old_actions_log_prob_batch: torch.Tensor,
        advantages_batch: torch.Tensor,
    ) -> torch.Tensor:
        # Keep PPO statistics in [B, 1] layout to avoid accidental [B, B]
        # broadcasting when log-prob tensors come from storage with a column dim.
        ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
        surrogate = -advantages_batch * ratio
        surrogate_clipped = -advantages_batch * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        return torch.max(surrogate, surrogate_clipped).mean()

    def _compute_value_loss(
        self,
        value_batch: torch.Tensor,
        returns_batch: torch.Tensor,
    ) -> torch.Tensor:
        # For adaptive KL schedule, we use the previous iteration's mean as target_values
        # This is a simplified version that just uses naive value loss
        return (returns_batch - value_batch).pow(2).mean()
