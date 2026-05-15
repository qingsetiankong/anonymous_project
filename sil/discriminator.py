from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import yaml

from rewards.sil_reward import compute_sil_reward
from rl.actor_critic_new import MLP


def _to_hidden_dims(hidden_dims: int | Sequence[int] | None) -> list[int]:
    """
    将隐藏层配置统一整理成 `list[int]`。

    输入:
    - `hidden_dims`:
      - `int`，例如 `256`
      - `Sequence[int]`，例如 `[512, 256]`
      - `None`，表示不使用隐藏层

    输出:
    - `list[int]`
      例如 `[256]`、`[512, 256]` 或 `[]`
    """
    if hidden_dims is None:
        return []
    if isinstance(hidden_dims, int):
        return [int(hidden_dims)]
    return [int(item) for item in hidden_dims]


def _as_optimizer_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """
    从优化器配置中提取可直接传给 PyTorch optimizer 的关键字参数。

    约定:
    - `name` 字段只用于选择优化器类，不会传入构造函数
    - 其余字段例如 `lr`、`weight_decay`、`betas` 会原样保留
    """
    return {key: value for key, value in config.items() if key != "name"}


def load_discriminator_config(config_path: str | Path) -> dict[str, Any]:
    """
    读取判别器 YAML 配置文件。

    输入:
    - `config_path`:
      YAML 文件路径，
      [configs/training/discriminator.yaml)

    输出:
    - `dict[str, Any]`
      至少可以包含以下顶层字段：
      - `model`
      - `optimizer`
      - `loss`

    说明:
    - 这里不会强制要求所有字段都存在
    - 缺失字段会在 `SILDiscriminator.from_config()` 中回退到默认值
    """
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise TypeError(f"配置文件内容必须是字典，实际得到: {type(config)}")
    return config


@dataclass
class DiscriminatorLossStats:
    """
    记录判别器训练时常用的统计量。

    字段:
    - `expert_score`:
      当前 batch 中 expert / SIL buffer 样本的平均判别器分数
    - `policy_score`:
      当前 batch 中 policy 样本的平均判别器分数
    - `expert_loss`:
      expert 分支的 MSE 损失
    - `policy_loss`:
      policy 分支的 MSE 损失
    - `gradient_penalty`:
      梯度惩罚项大小
    - `total_loss`:
      总损失
    """

    expert_score: float
    policy_score: float
    expert_loss: float
    policy_loss: float
    gradient_penalty: float
    total_loss: float


class SILDiscriminator(nn.Module):
    """
    PASIST / GASIL 中的自模仿判别器。

    判别器输入:
    - transition imitation observation，由连续两帧关节角拼接而成：
      1. `x_{t-1}` = `joint_pos_rel(12)`
      2. `x_t` = `joint_pos_rel(12)`
    - shape 常见为 `[batch_size, input_dim]`

    判别器输出:
    - 每个样本一个实值分数
    - shape 为 `[batch_size]`

    当前实现的重要特点:
    - 复用 [rl/actor_critic_new.py](/media/ubuntu20/D/robotic/复现/复现/rl/actor_critic_new.py) 中的 `MLP`
    - 支持从 YAML 配置文件初始化
    - 支持根据 YAML 自动构建优化器
    - 当前判别器只关注连续两帧关节角变化，便于更快收敛
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: int | Sequence[int] | None = (256, 256),
        hidden_activation: str = "relu",
        gradient_penalty_weight: float = 10.0,
        expert_target: float = 1.0,
        policy_target: float = -1.0,
    ) -> None:
        """
        输入:
        - `input_dim`:
          imitation observation 的特征维度
        - `hidden_dims`:
          隐藏层结构，支持：
          - 单个整数，例如 `256`
          - 列表，例如 `[512, 256]`
          - `None`
        - `gradient_penalty_weight`:
          梯度惩罚权重
        - `expert_target`:
          判别器希望 expert 样本接近的目标分数
        - `policy_target`:
          判别器希望 policy 样本接近的目标分数
        """
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = _to_hidden_dims(hidden_dims)
        self.hidden_activation = str(hidden_activation)
        self.gradient_penalty_weight = float(gradient_penalty_weight)
        self.expert_target = float(expert_target)
        self.policy_target = float(policy_target)

        # 直接复用 actor_critic_new 里的 MLP 实现，不再在本文件重复定义。
        self.backbone = MLP(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=1,
            output_activation=None,
            hidden_activation=self.hidden_activation,
        )

    @classmethod
    def from_config(cls, input_dim: int, config: dict[str, Any]) -> SILDiscriminator:
        """
        根据已经解析好的配置字典构造判别器。

        输入:
        - `input_dim`:
          imitation observation 的维度
        - `config`:
          来自 `load_discriminator_config()` 的字典

        支持的配置结构:
        ```yaml
        model:
          hidden_layers: [512, 256]
          activation: elu
        loss:
          gradient_penalty_weight: 10.0
          expert_target: 1.0
          policy_target: -1.0
        ```

        输出:
        - `SILDiscriminator`
        """
        model_cfg = dict(config.get("model", {}))
        loss_cfg = dict(config.get("loss", {}))
        hidden_dims = model_cfg.get("hidden_layers", model_cfg.get("hidden_dims", [256, 256]))
        hidden_activation = model_cfg.get("activation", model_cfg.get("hidden_activation", "relu"))

        return cls(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            hidden_activation=hidden_activation,
            gradient_penalty_weight=loss_cfg.get("gradient_penalty_weight", 10.0),
            expert_target=loss_cfg.get("expert_target", 1.0),
            policy_target=loss_cfg.get("policy_target", -1.0),
        )

    @classmethod
    def from_yaml(cls, input_dim: int, config_path: str | Path) -> SILDiscriminator:
        """
        直接从 YAML 文件构造判别器。

        输入:
        - `input_dim`:
          imitation observation 的维度
        - `config_path`:
          配置文件路径

        输出:
        - `SILDiscriminator`
        """
        config = load_discriminator_config(config_path)
        return cls.from_config(input_dim=input_dim, config=config)

    def build_optimizer(
        self,
        config: dict[str, Any] | None = None,
        config_path: str | Path | None = None,
    ) -> torch.optim.Optimizer:
        """
        根据配置构建判别器优化器。

        使用方式:
        1. 直接传配置字典
        2. 直接传 YAML 路径
        3. 两者都不传，此时使用默认 Adam(lr=1e-3)

        配置格式:
        ```yaml
        optimizer:
          name: Adam
          lr: 0.001
          weight_decay: 0.0
        ```

        输出:
        - `torch.optim.Optimizer`
        """
        if config is None and config_path is not None:
            config = load_discriminator_config(config_path)
        if config is None:
            config = {}

        optimizer_cfg = dict(config.get("optimizer", {}))
        optimizer_name = str(optimizer_cfg.get("name", "Adam"))
        optimizer_kwargs = _as_optimizer_kwargs(optimizer_cfg)
        if "lr" not in optimizer_kwargs:
            optimizer_kwargs["lr"] = 1.0e-3

        if not hasattr(torch.optim, optimizer_name):
            raise ValueError(f"不支持的优化器类型: {optimizer_name}")

        optimizer_cls = getattr(torch.optim, optimizer_name)
        return optimizer_cls(self.parameters(), **optimizer_kwargs)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        计算判别器分数。

        输入:
        - `observations`:
          `torch.Tensor`
          shape = `[batch_size, input_dim]`

        输出:
        - `torch.Tensor`
          shape = `[batch_size]`
        """
        scores = self.backbone(observations)
        return scores.squeeze(-1)

    def sil_reward(self, policy_samples: torch.Tensor, positive_margin: float = 0.0) -> torch.Tensor:
        """
        根据当前 policy 样本计算 SIL 奖励。

        输入:
        - `policy_samples`:
          当前策略采样到的 imitation observation
          shape = `[batch_size, input_dim]`

        输出:
        - `torch.Tensor`
          shape = `[batch_size]`
          值域由 `compute_sil_reward()` 约束在 `[0, 1]`
        """
        scores = self.forward(policy_samples)
        return compute_sil_reward(scores, positive_margin=positive_margin)

    def gradient_penalty(self, expert_samples: torch.Tensor) -> torch.Tensor:
        """
        计算针对 expert / SIL buffer 样本的梯度惩罚。

        输入:
        - `expert_samples`:
          来自 SIL buffer 的 imitation 样本
          shape = `[batch_size, input_dim]`

        输出:
        - `torch.Tensor`
          标量张量

        公式:
        - `E[||∇_x D(x)||^2]`
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
        gradient_penalty_weight: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        计算判别器总损失。

        输入:
        - `expert_samples`:
          expert / SIL buffer 样本
          shape = `[batch_size, input_dim]`
        - `policy_samples`:
          当前策略样本
          shape = `[batch_size, input_dim]`
        - `gradient_penalty_weight`:
          可选；如果传入则覆盖实例默认值

        输出:
        - `loss`:
          `torch.Tensor` 标量，可直接反向传播
        - `stats`:
          `dict[str, float]`，包含日志统计项：
          - `expert_score`
          - `policy_score`
          - `expert_loss`
          - `policy_loss`
          - `gradient_penalty`
          - `total_loss`
        """
        gp_weight = self.gradient_penalty_weight if gradient_penalty_weight is None else float(gradient_penalty_weight)

        expert_scores = self.forward(expert_samples)
        policy_scores = self.forward(policy_samples)

        expert_loss = torch.mean((expert_scores - self.expert_target) ** 2)
        policy_loss = torch.mean((policy_scores - self.policy_target) ** 2)
        penalty = self.gradient_penalty(expert_samples)

        loss = expert_loss + policy_loss + gp_weight * penalty
        stats = DiscriminatorLossStats(
            expert_score=float(expert_scores.mean().item()),
            policy_score=float(policy_scores.mean().item()),
            expert_loss=float(expert_loss.item()),
            policy_loss=float(policy_loss.item()),
            gradient_penalty=float(penalty.item()),
            total_loss=float(loss.item()),
        )
        return loss, stats.__dict__.copy()
