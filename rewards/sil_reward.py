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
    - discriminator_scores: 判别器输出分数，shape 可为 [batch] 或 [batch, 1]

    返回:
    - 与输入 shape 广播兼容的 torch.Tensor，值域约束在 [0, 1]
    """
    return torch.clamp(1.0 - 0.25 * (discriminator_scores - 1.0) ** 2, min=0.0, max=1.0)


def compute_sil_weight(mean_dtw_distance: float, sigma_sil: float, num_skills: int) -> float:
    """
    根据 DTW 统计值计算 SIL 奖励权重 omega_SIL。

    该实现是对论文 Eq. (7) 的一个工程化近似：
    - mean_dtw_distance 越小，说明 SIL buffer 中的高质量轨迹越接近目标姿态
    - 此时我们更愿意相信自模仿信号，因此 omega_SIL 会更大

    参数:
    - mean_dtw_distance: 当前 SIL buffer 中轨迹到目标姿态的平均 DTW 距离
    - sigma_sil: 控制缩放的超参数
    - num_skills: 技能数，用于做简单归一化，避免技能数变多时权重失控

    返回:
    - float 类型的 omega_SIL
    """
    normalized = max(float(mean_dtw_distance) - float(sigma_sil), 0.0) / max(int(num_skills), 1)
    return float(np.exp(-normalized))


def compute_mean_sil_dtw(summary_by_skill: dict[int, dict[str, float]]) -> float:
    """
    从 SIL buffer 的 summary 中提取全局平均 DTW / score 指标。

    注意:
    - 你的 `SILBuffer.summary()` 当前返回的是 `mean_score`
    - 如果你后面把 summary 改成同时包含 `mean_dtw`，这里可以直接切换

    参数:
    - summary_by_skill: 例如 `sil_buffer.summary()` 的输出

    返回:
    - 当前所有 skill 的平均 score；如果没有数据则返回 0.0
    """
    if not summary_by_skill:
        return 0.0
    values = [float(item.get("mean_score", 0.0)) for item in summary_by_skill.values()]
    return float(np.mean(values)) if values else 0.0
