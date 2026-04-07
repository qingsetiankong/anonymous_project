from __future__ import annotations

from typing import Any

import numpy as np


def _as_float_array(value: Any) -> np.ndarray:
    """
    将动作或状态统一转换成一维 float32 数组。
    """
    return np.asarray(value, dtype=np.float32).reshape(-1)


def action_magnitude_penalty(action: Any, weight: float = 1e-3) -> float:
    """
    惩罚动作幅值过大。

    直觉:
    - 动作过大通常意味着更激进的控制
    - 在机器人任务里往往会带来不稳定、能耗高、关节冲击大等问题

    返回值是负数或 0。
    """
    action_array = _as_float_array(action)
    return float(-float(weight) * np.square(action_array).sum())


def action_smoothness_penalty(action: Any, previous_action: Any | None, weight: float = 1e-3) -> float:
    """
    惩罚相邻时间步动作变化过快。

    如果没有 previous_action，则该项返回 0。
    """
    if previous_action is None:
        return 0.0

    current = _as_float_array(action)
    previous = _as_float_array(previous_action)
    if current.shape != previous.shape:
        raise ValueError("action 和 previous_action 的维度必须一致")

    return float(-float(weight) * np.square(current - previous).sum())


def posture_stability_penalty(
    observation: Any,
    reference_observation: Any | None = None,
    weight: float = 0.0,
) -> float:
    """
    一个可选的姿态稳定性惩罚项。

    当前默认权重是 0，目的是先保留接口。
    你后面接真实环境后，可以把 observation 中的 roll / pitch / height 等量拆出来，
    单独定义更合理的稳定性正则。
    """
    if reference_observation is None or weight == 0.0:
        return 0.0

    current = _as_float_array(observation)
    reference = _as_float_array(reference_observation)
    if current.shape != reference.shape:
        raise ValueError("observation 和 reference_observation 的维度必须一致")

    return float(-float(weight) * np.square(current - reference).mean())


def compute_regularization_reward(
    action: Any,
    previous_action: Any | None = None,
    observation: Any | None = None,
    reference_observation: Any | None = None,
    action_weight: float = 1e-3,
    smoothness_weight: float = 1e-3,
    stability_weight: float = 0.0,
) -> float:
    """
    计算论文中的 regularization reward r_R。

    这里使用三个常见正则项：
    - 动作幅值惩罚
    - 动作平滑惩罚
    - 可选的姿态稳定惩罚

    返回:
    - 标量正则奖励，通常为负数
    """
    reward = 0.0
    reward += action_magnitude_penalty(action=action, weight=action_weight)
    reward += action_smoothness_penalty(action=action, previous_action=previous_action, weight=smoothness_weight)
    reward += posture_stability_penalty(
        observation=observation,
        reference_observation=reference_observation,
        weight=stability_weight,
    )
    return float(reward)
