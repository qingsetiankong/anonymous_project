from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


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


def _to_tensor(value: Any, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if torch.is_tensor(value):
        if device is None:
            return value.to(dtype=dtype)
        return value.to(device=device, dtype=dtype)
    if device is None:
        return torch.as_tensor(value, dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device=device)


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

    def __init__(
        self,
        dtw_weight: float = 0.1,
        default_threshold: float = -np.inf,
        reference_length_ratio: float = 0.5,
        normalize_dtw: bool = True,
        dtw_feature_slices: list[tuple[int, int]] | None = None,
        task_reward_normalization_length: float = 128.0,
        min_task_return: float = -np.inf,
        max_dtw_distance: float = np.inf,
    ) -> None:
        """
        参数:
        - `dtw_weight`:
          DTW 在 assessment score 中的惩罚权重
        - `default_threshold`:
          各 skill 初始接纳阈值
        - `reference_length_ratio`:
          参考序列长度相对采样轨迹长度的比例
          PASIST 原文里 target pose 会扩展到 “half length”，因此默认设为 `0.5`
        - `normalize_dtw`:
          是否把 DTW 从累计代价转换为平均代价
        - `dtw_feature_slices`:
          可选；若提供，则 DTW 只使用这些切片对应的特征
          当前默认 `None`，表示直接使用完整 imitation observation
        - `task_reward_normalization_length`:
          计算 assessment score 时，用固定 horizon 对累计 task reward 做缩放。
          这样可以让 task reward 和已归一化的 DTW 保持更接近的量级，
          同时避免按实际轨迹长度归一化时过度偏向短片段。
        - `min_task_return`:
          轨迹进入 SIL buffer 前必须达到的最低累计 task reward。
          用于挡住训练初期明显不合格的轨迹。
        - `max_dtw_distance`:
          轨迹进入 SIL buffer 前允许的最大 DTW 距离。
          用于挡住和目标 keyframe 风格差异过大的轨迹。
        """
        self.dtw_weight = float(dtw_weight)
        self.default_threshold = float(default_threshold)
        self.reference_length_ratio = float(max(reference_length_ratio, 1e-6))
        self.normalize_dtw = bool(normalize_dtw)
        self.dtw_feature_slices = list(dtw_feature_slices) if dtw_feature_slices is not None else None
        self.task_reward_normalization_length = float(max(task_reward_normalization_length, 1.0))
        self.min_task_return = float(min_task_return)
        self.max_dtw_distance = float(max_dtw_distance)
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

    def _to_sequence_tensor(self, sequence, device: torch.device | None = None) -> torch.Tensor:
        """
        将轨迹序列统一转换为 shape = [T, D] 的二维张量。
        """
        if torch.is_tensor(sequence):
            tensor = _to_tensor(sequence, device=device, dtype=torch.float32)
        else:
            tensor = torch.stack(
                [_to_tensor(item, device=device, dtype=torch.float32).reshape(-1) for item in sequence],
                dim=0,
            )
        if tensor.ndim != 2:
            raise ValueError("轨迹序列必须能转换成二维张量 [T, D]")
        return tensor

    def _select_dtw_features(self, sequence_array: np.ndarray) -> np.ndarray:
        """
        从轨迹特征中提取参与 DTW 的子空间。

        默认行为:
        - 如果 `dtw_feature_slices is None`，直接使用完整特征
        - 如果提供了切片列表，就按这些切片拼接成新的 DTW 特征

        这样后面如果你想把 DTW 限定到 joint position，
        只需要在构造时传相应切片，而不用重写核心逻辑。
        """
        if self.dtw_feature_slices is None:
            return sequence_array

        features = [sequence_array[..., start:end] for start, end in self.dtw_feature_slices]
        if not features:
            raise ValueError("dtw_feature_slices 为空，无法构造 DTW 特征")
        return np.concatenate(features, axis=-1).astype(np.float32, copy=False)

    def _select_dtw_features_tensor(self, sequence_tensor: torch.Tensor) -> torch.Tensor:
        """
        从轨迹特征中提取参与 DTW 的子空间。
        """
        if self.dtw_feature_slices is None:
            return sequence_tensor

        features = [sequence_tensor[..., start:end] for start, end in self.dtw_feature_slices]
        if not features:
            raise ValueError("dtw_feature_slices 为空，无法构造 DTW 特征")
        return torch.cat(features, dim=-1)

    def replicate_target_pose(self, target_pose, trajectory_length: int) -> np.ndarray:
        """
        将单帧 target pose 复制成一段参考序列。

        按 PASIST 原文描述：
        - target pose 会沿时间轴重复
        - 参考序列长度取采样轨迹的 half length

        这里通过 `reference_length_ratio` 参数实现，
        默认值为 `0.5`。
        """
        reference_length = max(int(np.ceil(int(trajectory_length) * self.reference_length_ratio)), 1)
        target_pose = _to_numpy(target_pose, dtype=np.float32).reshape(1, -1)
        return np.repeat(target_pose, reference_length, axis=0)

    def replicate_target_pose_tensor(self, target_pose, trajectory_length: int, device: torch.device | None = None) -> torch.Tensor:
        """
        将单帧 target pose 复制成一段参考序列的张量版本。
        """
        reference_length = max(int(np.ceil(int(trajectory_length) * self.reference_length_ratio)), 1)
        target_pose_tensor = _to_tensor(target_pose, device=device, dtype=torch.float32).reshape(1, -1)
        return target_pose_tensor.repeat(reference_length, 1)

    def dtw_distance(self, sequence_a, sequence_b) -> float:
        """
        计算两段序列之间的 DTW 距离。

        距离定义:
        - 每个时间点之间的局部距离使用 L2 norm
        - 默认返回“平均对齐代价”，而不是原始累计代价
          这样可以减少轨迹长度变化带来的数值偏置
        """
        seq_a = self._select_dtw_features(self._to_sequence_array(sequence_a))
        seq_b = self._select_dtw_features(self._to_sequence_array(sequence_b))

        cost = np.full((len(seq_a) + 1, len(seq_b) + 1), np.inf, dtype=np.float32)
        path_length = np.full((len(seq_a) + 1, len(seq_b) + 1), np.inf, dtype=np.float32)
        cost[0, 0] = 0.0
        path_length[0, 0] = 0.0

        for index_a in range(1, len(seq_a) + 1):
            for index_b in range(1, len(seq_b) + 1):
                local_cost = np.linalg.norm(seq_a[index_a - 1] - seq_b[index_b - 1])
                predecessor_candidates = (
                    (cost[index_a - 1, index_b], path_length[index_a - 1, index_b]),
                    (cost[index_a, index_b - 1], path_length[index_a, index_b - 1]),
                    (cost[index_a - 1, index_b - 1], path_length[index_a - 1, index_b - 1]),
                )
                best_cost, best_steps = min(predecessor_candidates, key=lambda item: item[0])
                cost[index_a, index_b] = local_cost + best_cost
                path_length[index_a, index_b] = best_steps + 1.0

        total_cost = float(cost[len(seq_a), len(seq_b)])
        if not self.normalize_dtw:
            return total_cost

        total_steps = float(path_length[len(seq_a), len(seq_b)])
        if total_steps <= 0.0:
            return total_cost
        return total_cost / total_steps

    def dtw_distance_torch(self, sequence_a, sequence_b, device: torch.device | None = None) -> torch.Tensor:
        """
        计算两段序列之间的 DTW 距离的 torch 版本。
        """
        seq_a = self._select_dtw_features_tensor(self._to_sequence_tensor(sequence_a, device=device))
        seq_b = self._select_dtw_features_tensor(self._to_sequence_tensor(sequence_b, device=seq_a.device))

        len_a = int(seq_a.shape[0])
        len_b = int(seq_b.shape[0])
        inf = torch.tensor(float("inf"), dtype=torch.float32, device=seq_a.device)
        cost = torch.full((len_a + 1, len_b + 1), inf.item(), dtype=torch.float32, device=seq_a.device)
        path_length = torch.full((len_a + 1, len_b + 1), inf.item(), dtype=torch.float32, device=seq_a.device)
        cost[0, 0] = 0.0
        path_length[0, 0] = 0.0

        for index_a in range(1, len_a + 1):
            row = seq_a[index_a - 1].unsqueeze(0)
            local_costs = torch.linalg.norm(row - seq_b, dim=1)
            for index_b in range(1, len_b + 1):
                predecessor_costs = torch.stack(
                    [
                        cost[index_a - 1, index_b],
                        cost[index_a, index_b - 1],
                        cost[index_a - 1, index_b - 1],
                    ]
                )
                predecessor_steps = torch.stack(
                    [
                        path_length[index_a - 1, index_b],
                        path_length[index_a, index_b - 1],
                        path_length[index_a - 1, index_b - 1],
                    ]
                )
                best_index = torch.argmin(predecessor_costs)
                cost[index_a, index_b] = local_costs[index_b - 1] + predecessor_costs[best_index]
                path_length[index_a, index_b] = predecessor_steps[best_index] + 1.0

        total_cost = cost[len_a, len_b]
        if not self.normalize_dtw:
            return total_cost

        total_steps = path_length[len_a, len_b]
        if float(total_steps.item()) <= 0.0:
            return total_cost
        return total_cost / total_steps

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

        这里采用一个实用、容易调试且更贴近 PASIST 的实现：
        assessment_score = (task_return / normalization_length) - dtw_weight * dtw_distance

        直觉:
        - task reward 越高越好
        - 先按固定 horizon 归一化，避免累计回报天然压过 DTW
        - DTW 距离越小越好，因此作为惩罚项减去
        """
        if not trajectory.imitation_observations:
            raise ValueError("trajectory.imitation_observations 为空，无法计算 DTW")

        trajectory_sequence = self._to_sequence_array(trajectory.imitation_observations)
        reference_sequence = self.replicate_target_pose(target_pose=target_pose, trajectory_length=len(trajectory_sequence))
        dtw = self.dtw_distance(trajectory_sequence, reference_sequence)
        total_task_reward = self.task_return(trajectory)
        normalized_task_reward = total_task_reward / self.task_reward_normalization_length
        score = normalized_task_reward - self.dtw_weight * dtw
        return float(score), float(total_task_reward), float(dtw)

    def evaluate(self, trajectory: EpisodeTrajectory, target_pose) -> TrajectoryEvaluation:
        """
        评估一条轨迹，并判断其是否达到当前 skill 的接纳阈值。
        """
        score, total_task_reward, dtw = self.assessment_score(trajectory=trajectory, target_pose=target_pose)
        skill_id = int(trajectory.skill_id)
        passed_quality_gate = (
            total_task_reward >= self.min_task_return
            and dtw <= self.max_dtw_distance
        )
        accepted = passed_quality_gate and (score > self.best_score(skill_id))
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

    def evaluate_episode_tensor(self, episode: dict[str, Any], target_pose, device: torch.device | None = None) -> TrajectoryEvaluation:
        """
        直接基于 tensor episode 评估轨迹，避免中间转成 Python list / numpy。
        """
        imitation_sequence = episode.get("imitation_obs")
        if imitation_sequence is None:
            raise KeyError("episode 中缺少 imitation_obs，无法计算 DTW")
        imitation_sequence = self._to_sequence_tensor(imitation_sequence, device=device)
        if imitation_sequence.shape[0] == 0:
            raise ValueError("episode.imitation_obs 为空，无法计算 DTW")

        reward_task = episode.get("reward_task")
        if reward_task is None:
            raise KeyError("episode 中缺少 reward_task，无法计算 task return")
        reward_task_tensor = _to_tensor(reward_task, device=imitation_sequence.device, dtype=torch.float32).reshape(-1)

        skill_tensor = episode.get("skill_ids")
        if skill_tensor is None:
            raise KeyError("episode 中缺少 skill_ids，无法评估轨迹")
        skill_id = int(_to_tensor(skill_tensor, device=imitation_sequence.device, dtype=torch.long).reshape(-1)[0].item())

        reference_sequence = self.replicate_target_pose_tensor(
            target_pose=target_pose,
            trajectory_length=int(imitation_sequence.shape[0]),
            device=imitation_sequence.device,
        )
        dtw_tensor = self.dtw_distance_torch(imitation_sequence, reference_sequence, device=imitation_sequence.device)
        task_return_tensor = reward_task_tensor.sum()
        score_tensor = task_return_tensor / self.task_reward_normalization_length - self.dtw_weight * dtw_tensor

        dtw = float(dtw_tensor.item())
        task_return = float(task_return_tensor.item())
        score = float(score_tensor.item())
        passed_quality_gate = task_return >= self.min_task_return and dtw <= self.max_dtw_distance
        accepted = passed_quality_gate and (score > self.best_score(skill_id))
        if accepted:
            self._best_scores[skill_id] = score

        return TrajectoryEvaluation(
            skill_id=skill_id,
            task_return=task_return,
            dtw_distance=dtw,
            assessment_score=score,
            accepted=accepted,
            trajectory_length=int(imitation_sequence.shape[0]),
        )
