from __future__ import annotations

import torch


class RolloutStorage:
    """
    Stores transitions in [num_transitions_per_env, num_envs, ...] layout.

    Adapted from SLR's RolloutStorage: does NOT store critic_obs separately.
    During PPO update, critic uses the same observations as the actor.
    """

    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_std = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: list[int],
        privileged_obs_shape: list[int],
        action_shape: list[int],
        device: str = "cpu",
    ):
        self.device = device
        self.num_envs = int(num_envs)
        self.num_transitions_per_env = int(num_transitions_per_env)

        self.observations = torch.zeros(
            num_transitions_per_env, num_envs, *actor_obs_shape, device=device
        )
        self.critic_observations = torch.zeros(
            num_transitions_per_env, num_envs, *privileged_obs_shape, device=device
        )
        self.actions = torch.zeros(
            num_transitions_per_env, num_envs, *action_shape, device=device
        )
        self.rewards = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=device
        )
        self.dones = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=device
        )
        self.values = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=device
        )
        self.actions_log_prob = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=device
        )
        self.returns = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=device
        )
        self.advantages = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=device
        )
        self.mu = torch.zeros(
            num_transitions_per_env, num_envs, *action_shape, device=device
        )
        self.sigma = torch.zeros(
            num_transitions_per_env, num_envs, *action_shape, device=device
        )

        self._step = 0

    def add_transitions(self, transition: Transition):
        if self._step >= self.num_transitions_per_env:
            raise RuntimeError("RolloutStorage is full")
        self.observations[self._step].copy_(transition.observations)
        self.critic_observations[self._step].copy_(transition.critic_observations)
        self.actions[self._step].copy_(transition.actions)
        self.rewards[self._step].copy_(transition.rewards)
        self.dones[self._step].copy_(transition.dones.bool().unsqueeze(-1))
        self.values[self._step].copy_(transition.values)
        self.actions_log_prob[self._step].copy_(transition.actions_log_prob)
        self.mu[self._step].copy_(transition.action_mean)
        self.sigma[self._step].copy_(transition.action_std)
        self._step += 1

    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float):
        advantage = 0.0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.rewards[step]
                + next_is_not_terminal * gamma * next_values
                - self.values[step]
            )
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int):
        batch_size = self.num_envs * self.num_transitions_per_env
        num_mini_batches = max(int(num_mini_batches), 1)
        mini_batch_size = max(batch_size // num_mini_batches, 1)

        obs = self.observations.flatten(0, 1)
        critic_obs = self.critic_observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_log_prob = self.actions_log_prob.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for _ in range(num_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = min((i + 1) * mini_batch_size, batch_size)
                batch_indices = indices[start:end]

                yield (
                    obs[batch_indices],
                    critic_obs[batch_indices],
                    actions[batch_indices],
                    values[batch_indices],
                    advantages[batch_indices],
                    returns[batch_indices],
                    old_log_prob[batch_indices],
                    old_mu[batch_indices],
                    old_sigma[batch_indices],
                )

    def clear(self):
        self._step = 0

    @property
    def step(self) -> int:
        return self._step
