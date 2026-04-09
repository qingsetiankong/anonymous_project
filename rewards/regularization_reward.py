from __future__ import annotations

from typing import Any

import numpy as np


def _as_float_array(value: Any) -> np.ndarray:
    """
    将动作或状态统一转换成 `float32` numpy 数组。

    这里不主动展平，因为 regularization reward 需要兼容：
    - 单环境输入：shape = [D]
    - 并行环境输入：shape = [B, D]
    """
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return array.reshape(1)
    return array


def _sum_except_batch(array: np.ndarray) -> np.ndarray:
    """
    对“最后若干维是特征维”的数组求和。

    约定:
    - 一维输入 [D] 直接对全部特征求和，返回标量
    - 二维及以上输入 [B, ...] 对 batch 维之外的所有维度求和，返回 [B]
    """
    if array.ndim == 1:
        return np.asarray(array.sum(), dtype=np.float32)
    axes = tuple(range(1, array.ndim))
    return np.asarray(array.sum(axis=axes), dtype=np.float32)


def _mean_except_batch(array: np.ndarray) -> np.ndarray:
    """
    对 batch 维之外的所有维度求均值。
    """
    if array.ndim == 1:
        return np.asarray(array.mean(), dtype=np.float32)
    axes = tuple(range(1, array.ndim))
    return np.asarray(array.mean(axis=axes), dtype=np.float32)


def _maybe_scalar(value: np.ndarray | np.generic | float) -> float | np.ndarray:
    """
    单环境输入时返回 `float`，并行输入时返回 `np.ndarray`。
    """
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return array.astype(np.float32, copy=False)


def action_magnitude_penalty(action: Any, weight: float = 1e-3) -> float | np.ndarray:
    """
    惩罚动作幅值过大。

    直觉:
    - 动作过大通常意味着更激进的控制
    - 在机器人任务里往往会带来不稳定、能耗高、关节冲击大等问题

    输入:
    - `action`:
      - 单环境：shape = [D]
      - 并行环境：shape = [B, D]
    - `weight`:
      惩罚权重

    输出:
    - 单环境时返回 `float`
    - 并行环境时返回 shape = [B] 的 `np.ndarray`

    返回值总是负数或 0。
    """
    action_array = _as_float_array(action)
    penalty = -float(weight) * _sum_except_batch(np.square(action_array))
    return _maybe_scalar(penalty)


def action_smoothness_penalty(
    action: Any,
    previous_action: Any | None,
    weight: float = 1e-3,
) -> float | np.ndarray:
    """
    惩罚相邻时间步动作变化过快。

    输入:
    - `action`:
      当前动作，shape = [D] 或 [B, D]
    - `previous_action`:
      上一时刻动作，shape 必须与 `action` 一致
    - `weight`:
      惩罚权重

    输出:
    - 单环境时返回 `float`
    - 并行环境时返回 shape = [B] 的 `np.ndarray`

    如果没有 `previous_action`，该项返回 0。
    """
    if previous_action is None:
        return 0.0

    current = _as_float_array(action)
    previous = _as_float_array(previous_action)
    if current.shape != previous.shape:
        raise ValueError("action 和 previous_action 的维度必须一致")

    penalty = -float(weight) * _sum_except_batch(np.square(current - previous))
    return _maybe_scalar(penalty)


def posture_stability_penalty(
    observation: Any,
    reference_observation: Any | None = None,
    weight: float = 0.0,
) -> float | np.ndarray:
    """
    一个可选的姿态稳定性惩罚项。

    当前默认权重是 0，目的是先保留接口。
    你后面接真实环境后，可以把 observation 中的 roll / pitch / height 等量拆出来，
    单独定义更合理的稳定性正则。

    输入:
    - `observation`:
      当前观测，shape = [D] 或 [B, D]
    - `reference_observation`:
      参考观测，shape 必须与 `observation` 一致
    - `weight`:
      惩罚权重

    输出:
    - 单环境时返回 `float`
    - 并行环境时返回 shape = [B] 的 `np.ndarray`
    """
    if reference_observation is None or weight == 0.0:
        return 0.0

    current = _as_float_array(observation)
    reference = _as_float_array(reference_observation)
    if current.shape != reference.shape:
        raise ValueError("observation 和 reference_observation 的维度必须一致")

    penalty = -float(weight) * _mean_except_batch(np.square(current - reference))
    return _maybe_scalar(penalty)


def compute_regularization_reward(
    action: Any,
    previous_action: Any | None = None,
    observation: Any | None = None,
    reference_observation: Any | None = None,
    action_weight: float = 1e-3,
    smoothness_weight: float = 1e-3,
    stability_weight: float = 0.0,
) -> float | np.ndarray:
    """
    计算论文中的 regularization reward r_R。

    这里使用三个常见正则项：
    - 动作幅值惩罚
    - 动作平滑惩罚
    - 可选的姿态稳定惩罚

    返回:
    - 单环境时返回 `float`
    - 并行环境时返回 shape = [B] 的 `np.ndarray`

    说明:
    - 这个函数更偏“通用机器人控制正则项组合”
    - 适合你当前 PASIST 骨架阶段先跑通训练链
    - 后面接真实实验时，可以把这里替换成更贴合论文实现的 `r_R`
    """
    reward = np.asarray(action_magnitude_penalty(action=action, weight=action_weight), dtype=np.float32)
    reward = reward + np.asarray(
        action_smoothness_penalty(action=action, previous_action=previous_action, weight=smoothness_weight),
        dtype=np.float32,
    )
    reward = reward + np.asarray(
        posture_stability_penalty(
            observation=observation,
            reference_observation=reference_observation,
            weight=stability_weight,
        ),
        dtype=np.float32,
    )
    return _maybe_scalar(reward)
