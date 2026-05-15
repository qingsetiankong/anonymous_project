from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rl.actor_critic_new import MLP


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        init_std: float = 1.0,
        activation: str = "elu",
    ):
        super().__init__()
        self.mean_net = MLP(
            input_dim=obs_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
            output_activation=None,
            hidden_activation=activation,
        )
        self.std_param = nn.Parameter(
            torch.full((action_dim,), float(init_std), dtype=torch.float32)
        )
        self._distribution: Normal | None = None
        self._action_mean: torch.Tensor | None = None

    def _current_std(self) -> torch.Tensor:
        # `init_noise_std` in the config follows the legged_gym / RSL-RL
        # convention and represents the actual standard deviation, not log-std.
        return torch.clamp(self.std_param, min=1.0e-6)

    def _build_distribution(self, observations: torch.Tensor) -> Normal:
        mean = self.mean_net(observations)
        self._action_mean = mean
        std = self._current_std().expand_as(mean)
        dist = Normal(mean, std)
        self._distribution = dist
        return dist

    def sample(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._build_distribution(observations)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        return actions, log_probs

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self._distribution is None:
            raise RuntimeError("Must call sample() before get_actions_log_prob()")
        return self._distribution.log_prob(actions).sum(dim=-1, keepdim=True)

    def log_prob_and_entropy(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._build_distribution(observations)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1).mean()
        return log_probs, entropy

    @property
    def action_mean(self) -> torch.Tensor | None:
        return self._action_mean

    @property
    def action_std(self) -> torch.Tensor:
        return self._current_std()

    def get_std(self) -> torch.Tensor:
        return self._current_std()

    @property
    def entropy(self) -> torch.Tensor:
        if self._distribution is None:
            return torch.tensor(0.0)
        return self._distribution.entropy().sum(dim=-1).mean()

    def reset(self, dones=None):
        pass


class ValueFunction(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
    ):
        super().__init__()
        self.value_net = MLP(
            input_dim=obs_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            output_activation=None,
            hidden_activation=activation,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.value_net(observations)


class PasistActorCritic(nn.Module):
    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: tuple[int, ...] = (512, 256, 128),
        critic_hidden_dims: tuple[int, ...] = (512, 256, 128),
        init_noise_std: float = 1.0,
        activation: str = "elu",
    ):
        super().__init__()
        self.actor = GaussianPolicy(
            obs_dim=num_actor_obs,
            action_dim=num_actions,
            hidden_dims=actor_hidden_dims,
            init_std=float(init_noise_std),
            activation=activation,
        )
        self.critic = ValueFunction(
            obs_dim=num_critic_obs,
            hidden_dims=critic_hidden_dims,
            activation=activation,
        )

    def act(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor.sample(observations)[0]

    def evaluate(self, critic_observations: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_observations)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.actor.get_actions_log_prob(actions)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor.mean_net(observations)

    def train(self, mode: bool = True):
        super().train(mode)

    def test(self):
        self.eval()

    def reset(self, dones=None):
        pass
