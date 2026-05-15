from __future__ import annotations

import math
import unittest

import torch

from rewards.sil_reward import (
    compute_sil_confidence_weight,
    compute_sil_reward,
    compute_sil_warmup_weight,
    compute_sil_weight,
)


class SILRewardTests(unittest.TestCase):
    def test_compute_sil_reward_uses_positive_margin(self) -> None:
        scores = torch.tensor([-0.2, 0.0, 0.1, 0.55, 1.0], dtype=torch.float32)
        rewards = compute_sil_reward(scores, positive_margin=0.1)

        expected = torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(rewards, expected, atol=1.0e-6))

    def test_compute_sil_weight_only_penalizes_dtw_above_sigma(self) -> None:
        self.assertAlmostEqual(
            compute_sil_weight(mean_dtw_distance=0.55, sigma_sil=0.6, num_skills=1, dtw_decay_rate=10.0),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            compute_sil_weight(mean_dtw_distance=0.6, sigma_sil=0.6, num_skills=1, dtw_decay_rate=10.0),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            compute_sil_weight(mean_dtw_distance=0.65, sigma_sil=0.6, num_skills=1, dtw_decay_rate=10.0),
            math.exp(-0.5),
            places=6,
        )

    def test_compute_sil_confidence_weight_is_linear_between_margins(self) -> None:
        self.assertAlmostEqual(compute_sil_confidence_weight(score_margin=0.1, min_margin=0.2, max_margin=0.6), 0.0)
        self.assertAlmostEqual(compute_sil_confidence_weight(score_margin=0.4, min_margin=0.2, max_margin=0.6), 0.5)
        self.assertAlmostEqual(compute_sil_confidence_weight(score_margin=0.8, min_margin=0.2, max_margin=0.6), 1.0)

    def test_compute_sil_warmup_weight_ramps_after_buffer_ready(self) -> None:
        self.assertAlmostEqual(compute_sil_warmup_weight(expert_trajectory_count=10, warmup_start=20, warmup_trajectories=20), 0.0)
        self.assertAlmostEqual(compute_sil_warmup_weight(expert_trajectory_count=20, warmup_start=20, warmup_trajectories=20), 0.0)
        self.assertAlmostEqual(compute_sil_warmup_weight(expert_trajectory_count=30, warmup_start=20, warmup_trajectories=20), 0.5)
        self.assertAlmostEqual(compute_sil_warmup_weight(expert_trajectory_count=40, warmup_start=20, warmup_trajectories=20), 1.0)


if __name__ == "__main__":
    unittest.main()
