from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _to_numpy(value: Any, dtype=np.float32) -> np.ndarray:
    """
    将 torch / numpy / Python 标量统一整理为 numpy 数组。

    说明:
    - rollout_buffer.extract_episodes() 当前返回的是张量列表
    - 其中很多元素仍然位于 GPU 上
    - 因此这里必须先 `detach().cpu()`，再交给 `np.asarray`
    """
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=dtype)


@dataclass
class EpisodeTrajectory:
    """
    描述一条 episode 轨迹的结构化容器。

    这类对象通常在 rollout 结束后构造，用于：
    - 计算累计 task reward
    - 提取 imitation observation 做 DTW
    - 评估是否进入 SIL buffer
    """

    skill_id: int
    command: Any | None = None
    observations: list[Any] = field(default_factory=list)
    imitation_observations: list[Any] = field(default_factory=list)
    task_rewards: list[float] = field(default_factory=list)
    regularization_rewards: list[float] = field(default_factory=list)
    total_rewards: list[float] = field(default_factory=list)
    actions: list[Any] = field(default_factory=list)

    @property
    def length(self) -> int:
        """
        返回轨迹长度。
        """
        return len(self.imitation_observations) if self.imitation_observations else len(self.observations)


@dataclass
class TrajectoryEvaluation:
    """
    一条轨迹经过筛选器评估后的结果。
    """

    skill_id: int
    task_return: float
    dtw_distance: float
    assessment_score: float
    accepted: bool
    trajectory_length: int


class TrajectorySelector:
    """
    PASIST 中的高质量轨迹筛选器。

    职责:
    - 计算轨迹和 target pose 序列之间的 DTW 距离
    - 结合累计 task reward 形成 assessment score
    - 根据 score 判断轨迹是否应进入 SIL buffer

    这里采用“每个 skill 单独维护最佳分数阈值”的策略，
    避免某个高回报 skill 把其他 skill 完全压制掉。
    """

    def __init__(self, dtw_weight: float = 0.1, default_threshold: float = -np.inf) -> None:
        self.dtw_weight = float(dtw_weight)
        self.default_threshold = float(default_threshold)
        self._best_scores: dict[int, float] = {}

    def best_score(self, skill_id: int) -> float:
        """
        返回指定 skill 当前已知的最佳 assessment score。
        """
        return float(self._best_scores.get(int(skill_id), self.default_threshold))

    def _to_sequence_array(self, sequence) -> np.ndarray:
        """
        将轨迹序列统一转换为 shape = [T, D] 的二维数组。
        """
        array = np.asarray([_to_numpy(item, dtype=np.float32).reshape(-1) for item in sequence], dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("轨迹序列必须能转换成二维数组 [T, D]")
        return array

    def replicate_target_pose(self, target_pose, trajectory_length: int) -> np.ndarray:
        """
        将单帧 target pose 复制成一段参考序列。

        按论文描述，目标姿态会沿时间轴重复，
        用于和采样轨迹做 DTW 匹配。
        """
        trajectory_length = max(int(trajectory_length), 1)
        target_pose = _to_numpy(target_pose, dtype=np.float32).reshape(1, -1)
        return np.repeat(target_pose, trajectory_length, axis=0)

    def dtw_distance(self, sequence_a, sequence_b) -> float:
        """
        计算两段序列之间的 DTW 距离。

        距离定义:
        - 每个时间点之间的局部距离使用 L2 norm
        - 最终返回动态规划得到的最小累计匹配代价
        """
        seq_a = self._to_sequence_array(sequence_a)
        seq_b = self._to_sequence_array(sequence_b)

        cost = np.full((len(seq_a) + 1, len(seq_b) + 1), np.inf, dtype=np.float32)
        cost[0, 0] = 0.0

        for index_a in range(1, len(seq_a) + 1):
            for index_b in range(1, len(seq_b) + 1):
                local_cost = np.linalg.norm(seq_a[index_a - 1] - seq_b[index_b - 1])
                cost[index_a, index_b] = local_cost + min(
                    cost[index_a - 1, index_b],
                    cost[index_a, index_b - 1],
                    cost[index_a - 1, index_b - 1],
                )

        return float(cost[len(seq_a), len(seq_b)])

    def task_return(self, trajectory: EpisodeTrajectory) -> float:
        """
        计算轨迹累计 task reward。
        """
        if not trajectory.task_rewards:
            return 0.0
        return float(np.sum(_to_numpy(trajectory.task_rewards, dtype=np.float32)))

    def assessment_score(self, trajectory: EpisodeTrajectory, target_pose) -> tuple[float, float, float]:
        """
        计算轨迹评估分数。

        返回:
        - assessment_score
        - task_return
        - dtw_distance

        这里采用一个实用、容易调试的实现：
        assessment_score = task_return - dtw_weight * dtw_distance

        直觉:
        - task reward 越高越好
        - DTW 距离越小越好，因此作为惩罚项减去
        """
        if not trajectory.imitation_observations:
            raise ValueError("trajectory.imitation_observations 为空，无法计算 DTW")

        trajectory_sequence = self._to_sequence_array(trajectory.imitation_observations)
        reference_sequence = self.replicate_target_pose(target_pose=target_pose, trajectory_length=len(trajectory_sequence))
        dtw = self.dtw_distance(trajectory_sequence, reference_sequence)
        total_task_reward = self.task_return(trajectory)
        score = total_task_reward - self.dtw_weight * dtw
        return float(score), float(total_task_reward), float(dtw)

    def evaluate(self, trajectory: EpisodeTrajectory, target_pose) -> TrajectoryEvaluation:
        """
        评估一条轨迹，并判断其是否达到当前 skill 的接纳阈值。
        """
        score, total_task_reward, dtw = self.assessment_score(trajectory=trajectory, target_pose=target_pose)
        skill_id = int(trajectory.skill_id)
        accepted = score > self.best_score(skill_id)
        if accepted:
            self._best_scores[skill_id] = score

        return TrajectoryEvaluation(
            skill_id=skill_id,
            task_return=total_task_reward,
            dtw_distance=dtw,
            assessment_score=score,
            accepted=accepted,
            trajectory_length=trajectory.length,
        )

    def episode_from_buffer_dict(self, episode: dict[str, Any]) -> EpisodeTrajectory:
        """
        将 rollout_buffer.extract_episodes() 返回的 dict 转成 EpisodeTrajectory。

        这样可以让 trajectory selector 和 rollout buffer 保持解耦。
        """
        if "skill_ids" not in episode or not episode["skill_ids"]:
            raise KeyError("episode 中缺少 skill_ids，无法构造 EpisodeTrajectory")

        skill_id = int(_to_numpy(episode["skill_ids"][0], dtype=np.int64).reshape(-1)[0])
        return EpisodeTrajectory(
            skill_id=skill_id,
            observations=list(episode.get("obs", [])),
            imitation_observations=list(episode.get("imitation_obs", [])),
            task_rewards=[float(_to_numpy(item, dtype=np.float32).reshape(-1)[0]) for item in episode.get("reward_task", [])],
            regularization_rewards=[
                float(_to_numpy(item, dtype=np.float32).reshape(-1)[0]) for item in episode.get("reward_reg", [])
            ],
            total_rewards=[float(_to_numpy(item, dtype=np.float32).reshape(-1)[0]) for item in episode.get("reward_total", [])],
            actions=list(episode.get("actions", [])),
        )
