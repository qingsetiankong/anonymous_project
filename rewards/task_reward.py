from __future__ import annotations

from typing import Any

import numpy as np


def _as_float_array(value: Any) -> np.ndarray:
    """
    将输入统一转成一维 float32 数组。

    这样无论上游传进来的是 list、numpy 数组还是单个标量，
    reward 计算逻辑都可以统一处理。
    """
    return np.asarray(value, dtype=np.float32).reshape(-1)


def pose_tracking_reward(
    imitation_observation: Any,
    target_pose: Any,
    sigma: float = 1.0,
) -> float:
    """
    计算最基础的姿态跟踪奖励。

    设计思路:
    - imitation_observation 越接近 target_pose，奖励越高
    - 用指数函数把距离映射到 (0, 1]，便于和论文中的其它奖励项组合

    参数:
    - imitation_observation: 当前时刻用于模仿的观测子空间
    - target_pose: 当前 skill 对应的目标姿态
    - sigma: 距离缩放超参数

    返回:
    - 标量任务奖励，范围约为 (0, 1]
    """
    observation = _as_float_array(imitation_observation)
    target = _as_float_array(target_pose)
    if observation.shape != target.shape:
        raise ValueError("imitation_observation 和 target_pose 的维度必须一致")

    error = np.linalg.norm(observation - target)
    return float(np.exp(-error / max(float(sigma), 1e-6)))


def velocity_tracking_reward(
    commanded_velocity: float,
    measured_velocity: float,
    sigma: float = 0.25,
) -> float:
    """
    计算速度跟踪奖励。

    这个奖励项适合与你的 skill command 中的 `velocity` 分量联动，
    让策略不仅学会姿态，还学会以命令要求的速度执行该技能。
    """
    error = abs(float(commanded_velocity) - float(measured_velocity))
    return float(np.exp(-error / max(float(sigma), 1e-6)))


def command_consistency_reward(
    skill_id: int,
    command_skill_id: int,
) -> float:
    """
    一个简单的命令一致性奖励。

    如果当前样本所属的 skill 与 command 指定的 skill 一致，给 1.0；
    否则给 0.0。

    这个接口主要是为了在 mock 环境或早期复现阶段保留一个最小可用的
    skill-conditioned reward 信号。
    """
    return 1.0 if int(skill_id) == int(command_skill_id) else 0.0


def compute_task_reward(
    imitation_observation: Any,
    target_pose: Any,
    commanded_velocity: float | None = None,
    measured_velocity: float | None = None,
    skill_id: int | None = None,
    command_skill_id: int | None = None,
    pose_weight: float = 1.0,
    velocity_weight: float = 0.0,
    command_weight: float = 0.0,
    pose_sigma: float = 1.0,
    velocity_sigma: float = 0.25,
) -> float:
    """
    计算论文中的 task reward r_T。

    这不是论文里唯一可能的 reward 形式，而是一套和你当前代码结构兼容、
    适合逐步复现的组合版本：
    - 核心项: 姿态跟踪奖励
    - 可选项: 速度跟踪奖励
    - 可选项: command 一致性奖励

    参数:
    - imitation_observation / target_pose: 姿态匹配主信号
    - commanded_velocity / measured_velocity: 可选的速度命令跟踪信号
    - skill_id / command_skill_id: 可选的技能一致性信号
    - *_weight: 各个子奖励项的权重

    返回:
    - 标量任务奖励
    """
    reward = float(pose_weight) * pose_tracking_reward(
        imitation_observation=imitation_observation,
        target_pose=target_pose,
        sigma=pose_sigma,
    )

    if commanded_velocity is not None and measured_velocity is not None and velocity_weight != 0.0:
        reward += float(velocity_weight) * velocity_tracking_reward(
            commanded_velocity=commanded_velocity,
            measured_velocity=measured_velocity,
            sigma=velocity_sigma,
        )

    if skill_id is not None and command_skill_id is not None and command_weight != 0.0:
        reward += float(command_weight) * command_consistency_reward(
            skill_id=skill_id,
            command_skill_id=command_skill_id,
        )

    return float(reward)
