from __future__ import annotations

from typing import Any

import numpy as np


def _as_float_array(value: Any) -> np.ndarray:
    """
    将输入统一转成 `float32` numpy 数组。

    这里不强制展平成一维，原因是 task reward 后面需要同时支持：
    - 单样本输入：shape = [D]
    - 并行环境输入：shape = [B, D]

    参数:
    - value:
      任意可被 `np.asarray(..., dtype=np.float32)` 接受的对象，
      例如标量、list、numpy 数组等。

    返回:
    - `np.ndarray`
      dtype 为 `float32`，shape 尽量保持输入原状。
    """
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float32)


def _as_float_vector_or_batch(value: Any) -> np.ndarray:
    """
    将输入整理成“最后一维是特征维”的数组。

    约定:
    - 标量会被整理成 shape = [1]
    - 一维向量保持 shape = [D]
    - 二维及以上张量保持原 shape，不打乱 batch 维

    这个辅助函数主要给 `pose_tracking_reward()` 使用。
    """
    array = _as_float_array(value)
    if array.ndim == 0:
        return array.reshape(1)
    return array


def _maybe_scalar(value: np.ndarray | np.generic | float) -> float | np.ndarray:
    """
    将 0 维 numpy 结果转回 Python float。

    这样函数在单样本输入时返回 `float`，
    在 batched 输入时返回 `np.ndarray`，接口更直观。
    """
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return array.astype(np.float32, copy=False)


def pose_tracking_reward(
    imitation_observation: Any,
    target_pose: Any,
    sigma: float = 1.0,
) -> float | np.ndarray:
    """
    计算最基础的姿态跟踪奖励。

    设计思路:
    - imitation_observation 越接近 target_pose，奖励越高
    - 用指数函数把距离映射到 (0, 1]，便于和论文中的其它奖励项组合

    参数:
    - imitation_observation:
      当前时刻用于模仿的观测子空间。
      支持两种常见形状：
      - 单样本：shape = [D]
      - batched：shape = [B, D]
    - target_pose:
      当前 skill 对应的目标姿态。
      支持：
      - 单个目标姿态：shape = [D]
      - 与输入逐样本对应的 batched 目标：shape = [B, D]
    - sigma:
      距离缩放超参数，必须为正数。

    返回:
    - 如果输入是单样本，返回 `float`
    - 如果输入是 batched，返回 shape = [B] 的 `np.ndarray`

    值域:
    - 理论上在 `(0, 1]`
    """
    observation = _as_float_vector_or_batch(imitation_observation)
    target = _as_float_vector_or_batch(target_pose)
    if observation.shape != target.shape:
        # 允许 target_pose 是单帧姿态 [D]，自动广播到 batched 输入 [B, D]。
        if observation.ndim >= 2 and target.ndim == 1 and observation.shape[-1] == target.shape[-1]:
            target = np.broadcast_to(target, observation.shape)
        else:
            raise ValueError("imitation_observation 和 target_pose 的维度必须一致，或满足 [B, D] 对 [D] 广播")

    error = np.linalg.norm(observation - target, axis=-1 if observation.ndim >= 2 else 0)
    reward = np.exp(-error / max(float(sigma), 1e-6)).astype(np.float32, copy=False)
    return _maybe_scalar(reward)


def velocity_tracking_reward(
    commanded_velocity: float | np.ndarray,
    measured_velocity: float | np.ndarray,
    sigma: float = 0.25,
) -> float | np.ndarray:
    """
    计算速度跟踪奖励。

    这个奖励项适合与你的 skill command 中的 `velocity` 分量联动，
    让策略不仅学会姿态，还学会以命令要求的速度执行该技能。

    输入:
    - commanded_velocity:
      期望速度，可以是单个标量，也可以是 shape = [B] 的数组。
    - measured_velocity:
      实际测得速度，可以是单个标量，也可以是 shape = [B] 的数组。
    - sigma:
      距离缩放超参数，必须为正数。

    输出:
    - 单样本时返回 `float`
    - batched 时返回 shape = [B] 的 `np.ndarray`
    """
    commanded = _as_float_array(commanded_velocity)
    measured = _as_float_array(measured_velocity)
    error = np.abs(commanded - measured)
    reward = np.exp(-error / max(float(sigma), 1e-6)).astype(np.float32, copy=False)
    return _maybe_scalar(reward)


def command_consistency_reward(
    skill_id: int | np.ndarray,
    command_skill_id: int | np.ndarray,
) -> float | np.ndarray:
    """
    一个简单的命令一致性奖励。

    如果当前样本所属的 skill 与 command 指定的 skill 一致，给 1.0；
    否则给 0.0。

    这个接口主要是为了在 mock 环境或早期复现阶段保留一个最小可用的
    skill-conditioned reward 信号。

    输入:
    - `skill_id` / `command_skill_id`:
      可以是单个整数，也可以是 shape = [B] 的整数数组

    输出:
    - 单样本时返回 `float`
    - batched 时返回 shape = [B] 的 `np.ndarray`
    """
    current_skill = np.asarray(skill_id)
    command_skill = np.asarray(command_skill_id)
    reward = (current_skill == command_skill).astype(np.float32, copy=False)
    return _maybe_scalar(reward)


def compute_task_reward(
    imitation_observation: Any,
    target_pose: Any,
    commanded_velocity: float | np.ndarray | None = None,
    measured_velocity: float | np.ndarray | None = None,
    skill_id: int | np.ndarray | None = None,
    command_skill_id: int | np.ndarray | None = None,
    pose_weight: float = 1.0,
    velocity_weight: float = 0.0,
    command_weight: float = 0.0,
    pose_sigma: float = 1.0,
    velocity_sigma: float = 0.25,
) -> float | np.ndarray:
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

    输入形状:
    - 单样本模式：
      - `imitation_observation`: [D]
      - `target_pose`: [D]
      - 其他量为标量
    - batched 模式：
      - `imitation_observation`: [B, D]
      - `target_pose`: [D] 或 [B, D]
      - 其他量为标量或 [B]

    返回:
    - 单样本输入时返回 `float`
    - batched 输入时返回 shape = [B] 的 `np.ndarray`

    说明:
    - 这版实现已经兼容 Isaac Lab 并行环境常见的 batched 输入
    - 如果上游只处理单环境，你仍然会得到熟悉的标量输出
    """
    reward = np.asarray(
        float(pose_weight)
        * pose_tracking_reward(
            imitation_observation=imitation_observation,
            target_pose=target_pose,
            sigma=pose_sigma,
        ),
        dtype=np.float32,
    )

    if commanded_velocity is not None and measured_velocity is not None and velocity_weight != 0.0:
        reward = reward + np.asarray(
            float(velocity_weight)
            * velocity_tracking_reward(
                commanded_velocity=commanded_velocity,
                measured_velocity=measured_velocity,
                sigma=velocity_sigma,
            ),
            dtype=np.float32,
        )

    if skill_id is not None and command_skill_id is not None and command_weight != 0.0:
        reward = reward + np.asarray(
            float(command_weight)
            * command_consistency_reward(
                skill_id=skill_id,
                command_skill_id=command_skill_id,
            ),
            dtype=np.float32,
        )

    return _maybe_scalar(reward)
