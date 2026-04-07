from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from rewards.sil_reward import compute_sil_reward


def _to_hidden_dims(hidden_dims: int | Sequence[int]) -> list[int]:
    """
    将隐藏层配置统一转换成 list[int]。

    这样调用方既可以传单个整数，也可以传多个隐藏层宽度组成的序列。
    """
    if isinstance(hidden_dims, int):
        return [hidden_dims]
    return list(hidden_dims)


class MLP(nn.Module):
    """
    一个简单的多层感知机，用作 SIL 判别器主干网络。

    输入通常是 imitation observation，例如：
    - 关节位置
    - 姿态子空间特征
    - 由轨迹中抽取出的模仿观测
    """

    def __init__(self, input_dim: int, hidden_dims: int | Sequence[int], output_dim: int) -> None:
        super().__init__()
        dims = [input_dim, *_to_hidden_dims(hidden_dims), output_dim]
        layers: list[nn.Module] = []

        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
        - x: shape = [batch_size, input_dim]

        返回:
        - shape = [batch_size, output_dim]
        """
        return self.network(x)


@dataclass
class DiscriminatorLossStats:
    """
    记录判别器训练时常用的统计量，便于日志打印和调试。
    """

    expert_score: float
    policy_score: float
    gradient_penalty: float


class SILDiscriminator(nn.Module):
    """
    PASIST / GASIL 中的自模仿判别器。

    目标:
    - 对来自 SIL buffer 的“高质量轨迹片段”打高分
    - 对当前 policy 生成的轨迹片段打低分

    这里的判别器输出是一个实值分数，不是概率。
    对应论文中的训练目标，专家样本期望接近 +1，policy 样本期望接近 -1。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: int | Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.backbone = MLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        计算判别器分数。

        参数:
        - observations: shape = [batch_size, input_dim]

        返回:
        - shape = [batch_size]
        """
        scores = self.backbone(observations)
        return scores.squeeze(-1)

    def sil_reward(self, policy_samples: torch.Tensor) -> torch.Tensor:
        """
        根据当前 policy 样本计算 SIL 奖励。

        这里直接复用 rewards/sil_reward.py 中的公式：
        r_SIL = max(0, 1 - 0.25 * (D(x) - 1)^2)

        参数:
        - policy_samples: 当前策略采样到的 imitation observation

        返回:
        - shape = [batch_size]
        """
        scores = self.forward(policy_samples)
        return compute_sil_reward(scores)

    def gradient_penalty(self, expert_samples: torch.Tensor) -> torch.Tensor:
        """
        计算针对 expert / SIL buffer 样本的梯度惩罚。

        这里采用较稳定、实现简单的二范数惩罚：
        penalty = E[||∇_x D(x)||^2]

        参数:
        - expert_samples: 来自 SIL buffer 的模仿样本，shape = [batch_size, input_dim]

        返回:
        - 标量张量
        """
        expert_samples = expert_samples.detach().requires_grad_(True)
        scores = self.forward(expert_samples)
        gradients = torch.autograd.grad(
            outputs=scores.sum(),
            inputs=expert_samples,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.reshape(gradients.shape[0], -1)
        return gradients.pow(2).sum(dim=1).mean()

    def compute_loss(
        self,
        expert_samples: torch.Tensor,
        policy_samples: torch.Tensor,
        gradient_penalty_weight: float = 10.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        计算判别器总损失。

        损失组成:
        - expert loss: 希望 expert / SIL buffer 样本分数接近 +1
        - policy loss: 希望当前 policy 样本分数接近 -1
        - gradient penalty: 稳定训练

        参数:
        - expert_samples: shape = [batch_size, input_dim]
        - policy_samples: shape = [batch_size, input_dim]
        - gradient_penalty_weight: 梯度惩罚系数

        返回:
        - loss: 标量损失
        - stats: 便于日志记录的统计字典
        """
        expert_scores = self.forward(expert_samples)
        policy_scores = self.forward(policy_samples)

        expert_loss = torch.mean((expert_scores - 1.0) ** 2)
        policy_loss = torch.mean((policy_scores + 1.0) ** 2)
        penalty = self.gradient_penalty(expert_samples)

        loss = expert_loss + policy_loss + gradient_penalty_weight * penalty
        stats = {
            "expert_score": float(expert_scores.mean().item()),
            "policy_score": float(policy_scores.mean().item()),
            "gradient_penalty": float(penalty.item()),
        }
        return loss, stats
