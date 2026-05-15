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


def joint_acceleration_penalty(
    joint_velocity: Any | None,
    previous_joint_velocity: Any | None = None,
    weight: float = 0.0,
) -> float | np.ndarray:
    """
    惩罚关节速度在相邻控制步之间变化过快。

    说明:
    - KiRAS Table IV 中包含 `joint acceleration`
    - 当前工程里还没有单独把物理关节加速度稳定暴露出来
    - 因此这里采用一个离散近似：用相邻时刻的 joint velocity 差分
      作为 acceleration proxy

    输入:
    - `joint_velocity`:
      当前关节速度，shape = [D] 或 [B, D]
    - `previous_joint_velocity`:
      上一时刻关节速度，shape 必须与 `joint_velocity` 一致
    - `weight`:
      惩罚权重
    """
    if joint_velocity is None or previous_joint_velocity is None or weight == 0.0:
        return 0.0

    current = _as_float_array(joint_velocity)
    previous = _as_float_array(previous_joint_velocity)
    if current.shape != previous.shape:
        raise ValueError("joint_velocity 和 previous_joint_velocity 的维度必须一致")

    penalty = -float(weight) * _sum_except_batch(np.square(current - previous))
    return _maybe_scalar(penalty)


def roll_pitch_rate_penalty(
    base_angular_velocity: Any | None,
    weight: float = 0.0,
) -> float | np.ndarray:
    """
    惩罚 base 的 roll / pitch 角速度过大。

    对应 KiRAS Table IV 中更偏稳定性的角速度正则思想。
    这里默认使用 base angular velocity 的前两维 `[wx, wy]`。
    """
    if base_angular_velocity is None or weight == 0.0:
        return 0.0

    angular_velocity = _as_float_array(base_angular_velocity)
    if angular_velocity.shape[-1] < 2:
        raise ValueError("base_angular_velocity 至少需要包含 x/y 两个分量")

    penalty = -float(weight) * _sum_except_batch(np.square(angular_velocity[..., :2]))
    return _maybe_scalar(penalty)


def yaw_rate_penalty(
    base_angular_velocity: Any | None,
    weight: float = 0.0,
) -> float | np.ndarray:
    """
    惩罚 base 的 yaw 角速度过大。

    说明:
    - KiRAS Table IV 中给出了单独的 angular velocity z 正则项
    - 对当前项目，这一项默认建议小权重使用，避免过早压制转向能力
    """
    if base_angular_velocity is None or weight == 0.0:
        return 0.0

    angular_velocity = _as_float_array(base_angular_velocity)
    if angular_velocity.shape[-1] < 3:
        raise ValueError("base_angular_velocity 至少需要包含 z 分量")

    penalty = -float(weight) * _sum_except_batch(np.square(angular_velocity[..., 2:3]))
    return _maybe_scalar(penalty)


def lateral_velocity_penalty(
    lateral_velocity: Any | None,
    weight: float = 0.0,
) -> float | np.ndarray:
    """
    惩罚基座横向速度过大。

    对应 KiRAS Table IV 中的 `linear velocity y` 项，
    用于减少不必要的侧向漂移。
    """
    if lateral_velocity is None or weight == 0.0:
        return 0.0

    velocity = _as_float_array(lateral_velocity)
    penalty = -float(weight) * np.square(velocity)
    return _maybe_scalar(penalty)


def compute_regularization_reward(
    action: Any,
    previous_action: Any | None = None,
    observation: Any | None = None,
    reference_observation: Any | None = None,
    action_weight: float = 1e-3,
    smoothness_weight: float = 1e-3,
    stability_weight: float = 0.0,
    joint_velocity: Any | None = None,
    previous_joint_velocity: Any | None = None,
    joint_acceleration_weight: float = 0.0,
    base_angular_velocity: Any | None = None,
    roll_pitch_rate_weight: float = 0.0,
    yaw_rate_weight: float = 0.0,
    lateral_velocity: Any | None = None,
    lateral_velocity_weight: float = 0.0,
) -> float | np.ndarray:
    """
    计算论文中的 regularization reward r_R。

    当前组合分两类：

    1. 现有基础正则项
    - 动作幅值惩罚
    - 动作平滑惩罚
    - 可选的姿态稳定惩罚

    2. 从 KiRAS Table IV 中借鉴、且当前工程可直接提取到的正则项
    - 关节速度差分近似的 `joint acceleration`
    - `roll / pitch angular velocity`
    - `angular velocity z`
    - `linear velocity y`

    返回:
    - 单环境时返回 `float`
    - 并行环境时返回 shape = [B] 的 `np.ndarray`

    说明:
    - 这个函数更偏“通用机器人控制正则项组合”
    - 当前只接入“在现有训练链路中能稳定拿到”的 KiRAS 风格项
    - 像 feet drag / feet contact force / torques / delta torques 这类项，
      后续等环境 wrapper 显式暴露对应物理量后再继续扩展更稳妥
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
    reward = reward + np.asarray(
        joint_acceleration_penalty(
            joint_velocity=joint_velocity,
            previous_joint_velocity=previous_joint_velocity,
            weight=joint_acceleration_weight,
        ),
        dtype=np.float32,
    )
    reward = reward + np.asarray(
        roll_pitch_rate_penalty(
            base_angular_velocity=base_angular_velocity,
            weight=roll_pitch_rate_weight,
        ),
        dtype=np.float32,
    )
    reward = reward + np.asarray(
        yaw_rate_penalty(
            base_angular_velocity=base_angular_velocity,
            weight=yaw_rate_weight,
        ),
        dtype=np.float32,
    )
    reward = reward + np.asarray(
        lateral_velocity_penalty(
            lateral_velocity=lateral_velocity,
            weight=lateral_velocity_weight,
        ),
        dtype=np.float32,
    )
    return _maybe_scalar(reward)
