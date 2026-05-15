from __future__ import annotations

import unittest

import numpy as np

from sil.trajectory_selector import EpisodeTrajectory, TrajectorySelector


def _build_trajectory(length: int, total_task_reward: float) -> EpisodeTrajectory:
    per_step_reward = float(total_task_reward) / max(int(length), 1)
    imitation_frame = np.zeros(13, dtype=np.float32)
    return EpisodeTrajectory(
        skill_id=0,
        imitation_observations=[imitation_frame.copy() for _ in range(int(length))],
        task_rewards=[per_step_reward for _ in range(int(length))],
    )


class TrajectorySelectorTests(unittest.TestCase):
    def test_assessment_score_uses_fixed_task_reward_horizon(self) -> None:
        selector = TrajectorySelector(
            dtw_weight=0.0,
            task_reward_normalization_length=128.0,
        )
        target_pose = np.zeros(13, dtype=np.float32)

        short_trajectory = _build_trajectory(length=32, total_task_reward=64.0)
        long_trajectory = _build_trajectory(length=64, total_task_reward=64.0)

        short_score, short_task_return, short_dtw = selector.assessment_score(short_trajectory, target_pose=target_pose)
        long_score, long_task_return, long_dtw = selector.assessment_score(long_trajectory, target_pose=target_pose)

        self.assertAlmostEqual(short_task_return, 64.0, places=6)
        self.assertAlmostEqual(long_task_return, 64.0, places=6)
        self.assertAlmostEqual(short_dtw, 0.0, places=6)
        self.assertAlmostEqual(long_dtw, 0.0, places=6)
        self.assertAlmostEqual(short_score, 0.5, places=6)
        self.assertAlmostEqual(long_score, 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
