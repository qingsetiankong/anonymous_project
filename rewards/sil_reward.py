from __future__ import annotations

import numpy as np
import torch


def compute_sil_reward(discriminator_scores: torch.Tensor) -> torch.Tensor:
    """
    根据判别器分数计算 SIL 奖励。

    对应论文 Eq. (3):
    r_SIL = max(0, 1 - 0.25 * (D(x) - 1)^2)

    设计直觉:
    - 当判别器认为当前 policy 样本更像 SIL buffer 中的高质量样本时，
      D(x) 会更接近 1，此时奖励更高
    - 奖励被截断在 [0, 1] 范围内，便于和其他奖励项组合

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
    return torch.clamp(1.0 - 0.25 * (discriminator_scores - 1.0) ** 2, min=0.0, max=1.0)


def compute_sil_weight(mean_dtw_distance: float, sigma_sil: float, num_skills: int) -> float:
    """
    根据 DTW 统计值计算 SIL 奖励权重 omega_SIL。

    按 PASIST 原文 Eq. (7) 的含义实现：

    omega_SIL = exp(-|mean_dtw_distance - sigma_sil|)

    这里的 `mean_dtw_distance` 约定为：
    - 已经对所有 skill 做过平均的全局 DTW 统计
    - 也就是等价于论文中的 `(1 / N_m) * sum_p E[dDTW(...)]`

    说明:
    - 原文这里同样是范数/绝对值形式
    - 对当前标量 mean DTW 的工程实现，等价使用
      `abs(mean_dtw_distance - sigma_sil)`
    - 因此 `omega_SIL` 会稳定落在 `(0, 1]`

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
    return float(np.exp(-abs(float(mean_dtw_distance) - float(sigma_sil))))


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
