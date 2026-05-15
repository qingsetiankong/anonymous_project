from __future__ import annotations

import unittest

import numpy as np

from envs.BasePasistEnv import BasePasistEnv, PasistCommand
from rl.ppo_trainer import PPOTrainer, PPOTrainerConfig


class DummyAsymmetricEnv(BasePasistEnv):
    def __init__(self) -> None:
        self.num_envs = 1
        self._current_command: PasistCommand | None = None
        self._step_count = 0

    @property
    def obs_dim(self) -> int:
        return 4

    @property
    def critic_obs_dim(self) -> int:
        return 6

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def imitation_obs_dim(self) -> int:
        return 14

    @property
    def num_skills(self) -> int:
        return 1

    def reset(
        self,
        command: PasistCommand | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, dict]:
        _ = seed
        self._step_count = 0
        self._current_command = command or self.sample_command()
        observation = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        critic_obs = np.asarray([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=np.float32)
        return observation, self.build_info(
            observation=observation,
            command=self._current_command,
            measured_velocity=0.0,
            extra={
                "critic_obs": critic_obs,
                "base_height": 0.5,
                "base_pitch": 0.0,
            },
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        _ = action
        self._step_count += 1
        assert self._current_command is not None

        observation = np.asarray(
            [
                1.0 + self._step_count,
                2.0 + self._step_count,
                3.0 + self._step_count,
                4.0 + self._step_count,
            ],
            dtype=np.float32,
        )
        critic_base = 20.0 + 10.0 * self._step_count
        critic_obs = np.asarray(
            [critic_base + offset for offset in range(self.critic_obs_dim)],
            dtype=np.float32,
        )
        terminated = self._step_count >= 2
        info = self.build_info(
            observation=observation,
            command=self._current_command,
            measured_velocity=0.0,
            extra={
                "critic_obs": critic_obs,
                "base_height": 0.5,
                "base_pitch": 0.0,
            },
        )
        return observation, 0.0, terminated, False, info

    def sample_command(self, skill_id: int | None = None) -> PasistCommand:
        return self.build_command(velocity=0.0, skill_id=0 if skill_id is None else int(skill_id))

    def get_target_pose(self, skill_id: int) -> np.ndarray:
        target_pose = np.zeros(self.imitation_obs_dim, dtype=np.float32)
        target_pose[-1] = float(skill_id)
        return target_pose

    def extract_imitation_observation(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            imitation_obs = np.zeros(self.imitation_obs_dim, dtype=np.float32)
            imitation_obs[0] = 0.5
            imitation_obs[1 : 1 + obs.shape[0]] = obs
            return imitation_obs

        imitation_obs = np.zeros((obs.shape[0], self.imitation_obs_dim), dtype=np.float32)
        imitation_obs[:, 0] = 0.5
        imitation_obs[:, 1 : 1 + obs.shape[1]] = obs
        return imitation_obs


class AsymmetricCriticTests(unittest.TestCase):
    def test_critic_uses_dedicated_observation_stream(self) -> None:
        env = DummyAsymmetricEnv()
        config = PPOTrainerConfig(
            num_steps=2,
            ppo_epochs=1,
            mini_batch_size=2,
            device="cpu",
            enable_sil=False,
            use_skill_selector=False,
            pose_weight=0.0,
            velocity_weight=0.0,
            command_weight=0.0,
            action_weight=0.0,
            smoothness_weight=0.0,
        )
        trainer = PPOTrainer(env=env, config=config)

        self.assertEqual(trainer.policy.mean_net.layers[0].in_features, env.obs_dim)
        self.assertEqual(trainer.value_function.value_net.layers[0].in_features, env.critic_obs_dim)
        self.assertEqual(trainer.rollout_buffer.critic_obs.shape[-1], env.critic_obs_dim)

        trainer._reset_env_with_new_command(seed=0)
        trainer.collect_rollout()

        stored_critic_obs = trainer.rollout_buffer.critic_obs[: trainer.rollout_buffer.step, 0].cpu().numpy()
        np.testing.assert_allclose(
            stored_critic_obs[0],
            np.asarray([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            stored_critic_obs[1],
            np.asarray([30.0, 31.0, 32.0, 33.0, 34.0, 35.0], dtype=np.float32),
        )

        stats = trainer.update_policy()
        self.assertIn("ppo_value_loss", stats)


if __name__ == "__main__":
    unittest.main()
