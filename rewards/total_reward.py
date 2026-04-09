from __future__ import annotations

import math
from typing import Any

import numpy as np


def _as_float_array(value: Any) -> np.ndarray:
    """
    将输入统一转成 float32 numpy 数组。

    这个辅助函数主要是为了兼容：
    - 单环境标量 reward
    - 并行环境下的 batched reward 向量
    """
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float32)


def _maybe_scalar(value: np.ndarray | np.generic | float) -> float | np.ndarray:
    """
    单环境时返回 float，并行环境时返回 numpy 数组。
    """
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return array.astype(np.float32, copy=False)


def compute_task_weight(task_reward: float | np.ndarray, sigma_t: float) -> float | np.ndarray:
    """
    计算论文中的 omega_T。

    当前采用一个更稳健的工程近似：
    omega_T = exp(-max(r_T - sigma_T, 0))

    直觉:
    - 当 `task_reward <= sigma_t` 时，认为任务还没有学稳，此时 `omega_T = 1`
    - 当 `task_reward > sigma_t` 时，`omega_T` 会随任务奖励增大而衰减
    - 这样可以保证 `omega_T` 始终落在 `(0, 1]` 内，
      避免出现大于 1 导致 `(1 - omega_T)` 变成负数的问题

    输入:
    - `task_reward`:
      单个标量，或 shape = [B] 的 batched 任务奖励
    - `sigma_t`:
      控制切换速度的超参数

    输出:
    - 单环境时返回 `float`
    - batched 时返回 shape = [B] 的 `np.ndarray`
    """
    task_reward_array = _as_float_array(task_reward)
    clipped_gap = np.maximum(task_reward_array - float(sigma_t), 0.0)
    omega_t = np.exp(-clipped_gap, dtype=np.float32)
    return _maybe_scalar(omega_t)


def compute_total_reward(
    task_reward: float | np.ndarray,
    sil_reward: float | np.ndarray,
    regularization_reward: float | np.ndarray,
    omega_t: float | np.ndarray,
    omega_sil: float | np.ndarray,
    omega_r: float = 1.0,
) -> float | np.ndarray:
    """
    计算论文中的总奖励 r。

    对应 Eq. (6):
    r = omega_SIL * omega_T * r_SIL + (1 - omega_T) * r_T + omega_R * r_R

    参数:
    - task_reward: r_T
    - sil_reward: r_SIL
    - regularization_reward: r_R
    - omega_t: task reward 的动态权重
    - omega_sil: SIL reward 的动态权重
    - omega_r: 正则项权重，论文中固定为 1.0

    输入形状:
    - 可以全是标量
    - 也可以是彼此可广播的 batched 向量，例如 shape = [B]

    输出:
    - 单环境时返回 `float`
    - batched 时返回 shape = [B] 的 `np.ndarray`
    """
    task_reward_array = _as_float_array(task_reward)
    sil_reward_array = _as_float_array(sil_reward)
    regularization_reward_array = _as_float_array(regularization_reward)
    omega_t_array = _as_float_array(omega_t)
    omega_sil_array = _as_float_array(omega_sil)

    total_reward = (
        omega_sil_array * omega_t_array * sil_reward_array
        + (1.0 - omega_t_array) * task_reward_array
        + float(omega_r) * regularization_reward_array
    )
    return _maybe_scalar(total_reward)


def compute_reward_terms(
    task_reward: float | np.ndarray,
    sil_reward: float | np.ndarray,
    regularization_reward: float | np.ndarray,
    sigma_t: float,
    omega_sil: float | np.ndarray,
    omega_r: float = 1.0,
) -> dict[str, float | np.ndarray]:
    """
    一次性返回 reward 组合时常用的所有中间量。

    适合 trainer 里做日志记录，避免手动重复计算。

    输入:
    - `task_reward` / `sil_reward` / `regularization_reward`:
      标量或 batched 向量
    - `sigma_t`:
      计算 `omega_t` 的超参数
    - `omega_sil`:
      标量或 batched 向量
    - `omega_r`:
      正则项权重

    输出:
    - 一个字典，包含：
      - `task_reward`
      - `sil_reward`
      - `regularization_reward`
      - `omega_t`
      - `omega_sil`
      - `omega_r`
      - `total_reward`

    每个值在单环境时是 `float`，在 batched 模式下是 `np.ndarray`。
    """
    omega_t = compute_task_weight(task_reward=task_reward, sigma_t=sigma_t)
    total_reward = compute_total_reward(
        task_reward=task_reward,
        sil_reward=sil_reward,
        regularization_reward=regularization_reward,
        omega_t=omega_t,
        omega_sil=omega_sil,
        omega_r=omega_r,
    )
    return {
        "task_reward": _maybe_scalar(_as_float_array(task_reward)),
        "sil_reward": _maybe_scalar(_as_float_array(sil_reward)),
        "regularization_reward": _maybe_scalar(_as_float_array(regularization_reward)),
        "omega_t": _maybe_scalar(_as_float_array(omega_t)),
        "omega_sil": _maybe_scalar(_as_float_array(omega_sil)),
        "omega_r": float(omega_r),
        "total_reward": _maybe_scalar(_as_float_array(total_reward)),
    }
