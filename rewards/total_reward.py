from __future__ import annotations

import math


def compute_task_weight(task_reward: float, sigma_t: float) -> float:
    """
    计算论文中的 omega_T。

    对应论文 Eq. (8) 的工程实现近似：
    omega_T = exp(-(r_T - sigma_T))

    直觉:
    - 当 task reward 还不够高时，omega_T 会更大，
      使训练更偏向 imitation / exploitation
    - 当 task reward 已经较好时，omega_T 会减小，
      从而让总奖励更依赖 task reward 本身
    """
    return float(math.exp(-(float(task_reward) - float(sigma_t))))


def compute_total_reward(
    task_reward: float,
    sil_reward: float,
    regularization_reward: float,
    omega_t: float,
    omega_sil: float,
    omega_r: float = 1.0,
) -> float:
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
    """
    return float(
        float(omega_sil) * float(omega_t) * float(sil_reward)
        + (1.0 - float(omega_t)) * float(task_reward)
        + float(omega_r) * float(regularization_reward)
    )


def compute_reward_terms(
    task_reward: float,
    sil_reward: float,
    regularization_reward: float,
    sigma_t: float,
    omega_sil: float,
    omega_r: float = 1.0,
) -> dict[str, float]:
    """
    一次性返回 reward 组合时常用的所有中间量。

    适合 trainer 里做日志记录，避免手动重复计算。
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
        "task_reward": float(task_reward),
        "sil_reward": float(sil_reward),
        "regularization_reward": float(regularization_reward),
        "omega_t": float(omega_t),
        "omega_sil": float(omega_sil),
        "omega_r": float(omega_r),
        "total_reward": float(total_reward),
    }
