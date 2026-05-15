from __future__ import annotations

import numpy as np
import torch


def compute_sil_reward(
    discriminator_scores: torch.Tensor,
    positive_margin: float = 0.0,
) -> torch.Tensor:
    """
    根据判别器分数计算 SIL 奖励。

    当前实现采用单边阈值化形式：

    r_SIL = clamp((D(x) - positive_margin) / (1 - positive_margin), 0, 1)

    设计直觉:
    - 只有当判别器明确给出“更像 expert”的正分数时，SIL 才开始生效
    - `positive_margin > 0` 时，可以进一步抑制 `D(x) ≈ 0` 的模糊区奖励
    - 奖励仍然被约束在 `[0, 1]`，便于和其他奖励项组合

    参数:
    - `discriminator_scores`:
      判别器输出分数，通常为 `torch.Tensor`
      常见 shape:
      - `[batch]`
      - `[batch, 1]`

    返回:
    - `torch.Tensor`
      shape 与输入广播兼容，值域约束在 `[0, 1]`
    """
    margin = float(positive_margin)
    denom = max(1.0 - margin, 1.0e-6)
    normalized = (discriminator_scores - margin) / denom
    return torch.clamp(normalized, min=0.0, max=1.0)


def compute_sil_weight(
    mean_dtw_distance: float,
    sigma_sil: float,
    num_skills: int,
    dtw_decay_rate: float = 1.0,
) -> float:
    """
    根据 DTW 统计值计算 SIL 奖励权重 omega_SIL。

    当前实现采用单调降权形式：

    omega_SIL = exp(-dtw_decay_rate * max(mean_dtw_distance - sigma_sil, 0))

    这里的 `mean_dtw_distance` 约定为：
    - 已经对所有 skill 做过平均的全局 DTW 统计
    - 也就是等价于论文中的 `(1 / N_m) * sum_p E[dDTW(...)]` 的工程聚合值

    说明:
    - 当 `mean_dtw_distance <= sigma_sil` 时，说明 buffer 质量已达到目标阈值，
      此时不额外降权，返回 1.0
    - 当 `mean_dtw_distance > sigma_sil` 时，距离越差，衰减越快
    - `dtw_decay_rate` 控制衰减速度，越大越严格

    因此，虽然函数签名中仍然保留 `num_skills` 参数以兼容现有调用链，
    但当前实现不会再次除以 `num_skills`，避免重复平均。

    参数:
    - `mean_dtw_distance`:
      当前 SIL buffer 中轨迹到目标姿态的平均 DTW 距离
    - `sigma_sil`:
      控制缩放的超参数
    - `num_skills`:
      为兼容旧接口而保留；当前公式实现中不再直接使用

    返回:
    - `float`
      即 `omega_SIL`

    额外约定:
    - 如果 `mean_dtw_distance` 不是有限数值，例如 `inf`，
      说明当前还没有可靠 DTW 统计，此时返回 `0.0`
    - 这样训练早期不会错误放大 SIL 奖励
    """
    if not np.isfinite(float(mean_dtw_distance)):
        return 0.0
    del num_skills
    excess_distance = max(float(mean_dtw_distance) - float(sigma_sil), 0.0)
    return float(np.exp(-float(dtw_decay_rate) * excess_distance))


def compute_sil_confidence_weight(
    score_margin: float,
    min_margin: float,
    max_margin: float,
) -> float:
    """
    根据判别器 expert-policy 分离度计算置信度权重。

    线性门控形式：

    omega_conf = clip((score_margin - min_margin) / (max_margin - min_margin), 0, 1)

    其中：
    - `score_margin = mean(D_expert) - mean(D_policy)`
    - 当 margin 小于 `min_margin`，视为判别器分离度不足，返回 0
    - 当 margin 大于 `max_margin`，视为判别器分离度足够，返回 1
    """
    min_margin_value = float(min_margin)
    max_margin_value = float(max_margin)
    if max_margin_value <= min_margin_value:
        return float(1.0 if float(score_margin) >= max_margin_value else 0.0)
    normalized = (float(score_margin) - min_margin_value) / (max_margin_value - min_margin_value)
    return float(np.clip(normalized, 0.0, 1.0))


def compute_sil_warmup_weight(
    expert_trajectory_count: int,
    warmup_start: int,
    warmup_trajectories: int,
) -> float:
    """
    根据当前 expert 轨迹数计算 SIL warmup 权重。

    形式：

    omega_warmup = clip((count - warmup_start) / warmup_trajectories, 0, 1)

    其中：
    - `warmup_start` 通常取判别器开始训练的最小 buffer 条数
    - `warmup_trajectories` 控制从“刚能训练”到“完全放开”还需要新增多少条 expert
    """
    count = int(max(expert_trajectory_count, 0))
    start = int(max(warmup_start, 0))
    span = int(max(warmup_trajectories, 0))
    if span <= 0:
        return float(1.0 if count >= start else 0.0)
    normalized = (count - start) / float(span)
    return float(np.clip(normalized, 0.0, 1.0))


def compute_mean_sil_dtw(summary_by_skill: dict[int, dict[str, float]]) -> float:
    """
    从 `SILBuffer.summary()` 的输出中提取全局平均 DTW。

    重要说明:
    - 这个函数现在只读取真正和 DTW 相关的字段，不再错误地把 `mean_score`
      当作 DTW 使用
    - 如果 summary 中没有任何 DTW 字段，则返回 `inf`
    - 配合 `compute_sil_weight()`，这会让 `omega_SIL = 0.0`
      从而在缺少可靠 DTW 统计时安全地关闭 SIL 权重

    参数:
    - `summary_by_skill`:
      例如 `sil_buffer.summary()` 的输出
      每个 skill 的 value 最好至少包含：
      - `mean_dtw`
      也兼容别名：
      - `dtw_mean`
      - `avg_dtw`

    返回:
    - `float`
      当前所有 skill 的平均 DTW；如果没有任何 DTW 统计则返回 `inf`
    """
    if not summary_by_skill:
        return float("inf")

    values: list[float] = []
    for item in summary_by_skill.values():
        for key in ("mean_dtw", "dtw_mean", "avg_dtw"):
            if key in item:
                values.append(float(item[key]))
                break

    if not values:
        return float("inf")
    return float(np.mean(values))
