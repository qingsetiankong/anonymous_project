from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SILTrajectory:
    """
    SIL buffer 中存储的一条高质量轨迹。

    字段说明:
    - skill_id: 该轨迹对应的技能类别
    - assessment_score: 轨迹质量分数，通常由 task reward + DTW 共同决定
    - imitation_observations: 用于判别器训练的模仿观测序列
    - trajectory: 原始轨迹附加信息，保留给后续分析或调试
    - metadata: 补充元数据，例如 command、episode_id、长度等
    """

    skill_id: int
    assessment_score: float
    imitation_observations: np.ndarray
    trajectory: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """
        返回该轨迹的时间长度。
        """
        return int(len(self.imitation_observations))


class SILBuffer:
    """
    按 skill 分桶存储高质量轨迹的自模仿缓存。

    设计目标:
    - 每个 skill 单独保留若干条最优轨迹，避免一种技能挤占全部容量
    - 采样时从所有 skill 的高质量轨迹中抽取 imitation observation
    - 为日志系统提供每个 skill 的统计摘要
    """

    def __init__(self, capacity_per_skill: int = 8, random_seed: int = 0) -> None:
        self.capacity_per_skill = int(capacity_per_skill)
        self._rng = np.random.default_rng(random_seed)
        self._storage: dict[int, list[SILTrajectory]] = {}

    def __len__(self) -> int:
        """
        返回 buffer 中轨迹条目的总数量，而不是总时间步数。
        """
        return sum(len(entries) for entries in self._storage.values())

    def skills(self) -> list[int]:
        """
        返回当前 buffer 中已有的 skill id 列表。
        """
        return sorted(self._storage.keys())

    def add(
        self,
        skill_id: int,
        imitation_observations: np.ndarray,
        assessment_score: float,
        trajectory: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        向指定 skill 的桶中加入一条高质量轨迹。

        保留策略:
        - 如果该 skill 还没满，直接加入
        - 如果已满，仅当新轨迹分数高于当前最差轨迹时替换

        返回:
        - True: 新轨迹被接纳
        - False: 新轨迹被丢弃
        """
        skill_id = int(skill_id)
        imitation_observations = np.asarray(imitation_observations, dtype=np.float32)
        if imitation_observations.ndim != 2:
            raise ValueError("imitation_observations 必须是二维数组，shape 应为 [T, imitation_obs_dim]")

        entry = SILTrajectory(
            skill_id=skill_id,
            assessment_score=float(assessment_score),
            imitation_observations=imitation_observations,
            trajectory=trajectory,
            metadata={} if metadata is None else dict(metadata),
        )

        bucket = self._storage.setdefault(skill_id, [])
        if len(bucket) < self.capacity_per_skill:
            bucket.append(entry)
            bucket.sort(key=lambda item: item.assessment_score, reverse=True)
            return True

        worst_index = min(range(len(bucket)), key=lambda index: bucket[index].assessment_score)
        if entry.assessment_score <= bucket[worst_index].assessment_score:
            return False

        bucket[worst_index] = entry
        bucket.sort(key=lambda item: item.assessment_score, reverse=True)
        return True

    def add_episode(
        self,
        episode: dict[str, Any],
        assessment_score: float,
        skill_id: int | None = None,
        imitation_key: str = "imitation_obs",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        从 extract_episodes() 产出的 episode 字典中提取信息并加入 buffer。

        参数:
        - episode: rollout_buffer.extract_episodes() 返回的单条 episode
        - assessment_score: 轨迹质量评估值
        - skill_id: 可选；若不提供则从 episode["skill_ids"] 的首元素推断
        - imitation_key: episode 中保存 imitation sequence 的键名
        - metadata: 附加信息
        """
        if imitation_key not in episode:
            raise KeyError(f"episode 中缺少 imitation 序列字段: {imitation_key}")

        if skill_id is None:
            if "skill_ids" not in episode or not episode["skill_ids"]:
                raise KeyError("无法从 episode 中推断 skill_id，请显式传入")
            skill_id = int(np.asarray(episode["skill_ids"][0]).reshape(-1)[0])

        imitation_observations = np.asarray(
            [np.asarray(item, dtype=np.float32).reshape(-1) for item in episode[imitation_key]],
            dtype=np.float32,
        )

        return self.add(
            skill_id=skill_id,
            imitation_observations=imitation_observations,
            assessment_score=assessment_score,
            trajectory=episode,
            metadata=metadata,
        )

    def all_entries(self) -> list[SILTrajectory]:
        """
        返回所有 skill 桶中的轨迹条目列表。
        """
        entries: list[SILTrajectory] = []
        for bucket in self._storage.values():
            entries.extend(bucket)
        return entries

    def sample(self, batch_size: int, skill_id: int | None = None) -> np.ndarray:
        """
        从 buffer 中采样若干个 imitation observation。

        采样逻辑:
        - 先选择可用轨迹池
        - 再在轨迹池中随机选一条轨迹
        - 最后在该轨迹的时间维度上随机选一个时刻

        参数:
        - batch_size: 返回样本数
        - skill_id: 若提供，则只从指定 skill 的桶中采样

        返回:
        - shape = [batch_size, imitation_obs_dim] 的 numpy 数组
        """
        entries = self._storage.get(int(skill_id), []) if skill_id is not None else self.all_entries()
        if not entries:
            raise ValueError("SILBuffer 为空，无法采样")

        samples = []
        for _ in range(int(batch_size)):
            entry = entries[int(self._rng.integers(len(entries)))]
            time_index = int(self._rng.integers(entry.length))
            samples.append(entry.imitation_observations[time_index])
        return np.asarray(samples, dtype=np.float32)

    def summary(self) -> dict[int, dict[str, float]]:
        """
        返回按 skill 聚合的统计摘要，便于训练日志使用。
        """
        result: dict[int, dict[str, float]] = {}
        for skill_id, bucket in self._storage.items():
            scores = np.asarray([entry.assessment_score for entry in bucket], dtype=np.float32)
            lengths = np.asarray([entry.length for entry in bucket], dtype=np.float32)
            result[int(skill_id)] = {
                "count": float(len(bucket)),
                "mean_score": float(scores.mean()) if len(scores) else 0.0,
                "max_score": float(scores.max()) if len(scores) else 0.0,
                "min_score": float(scores.min()) if len(scores) else 0.0,
                "mean_length": float(lengths.mean()) if len(lengths) else 0.0,
            }
        return result
