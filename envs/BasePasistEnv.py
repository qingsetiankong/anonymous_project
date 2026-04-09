from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PasistCommand:
    """
    PASIST 训练中使用的统一 command 数据结构。

    字段说明:
    - velocity:
      连续速度命令，对应论文中的 v，通常取值在 [-0.5, 0.5] 范围内。
    - skill_id:
      当前要执行或训练的技能编号。
    - skill_name:
      技能的可读名称，方便日志记录和调试。
    - one_hot:
      技能编号对应的 one-hot 向量，对应论文中的离散 motion command m。

    这个结构的作用是把“速度命令 + 技能命令”打包在一起，
    让环境、trainer、skill selector 之间的接口统一。
    """

    velocity: float
    skill_id: int
    skill_name: str
    one_hot: np.ndarray


class BasePasistEnv(ABC):
    """
    PASIST 项目的通用环境基类。

    这个基类不绑定具体仿真器，也不绑定 Gym / Isaac Gym / Mujoco 等框架，
    目的是先把“训练时必须依赖的环境接口”定义清楚。

    任何具体环境实现，无论是：
    - 早期调试用的 mock 环境
    - Isaac Gym 中的 quadruped 环境
    - 后续接真实机器人前的仿真封装
    都建议继承这个类。

    该基类服务于四类模块：
    1. PPO trainer:
       需要 reset / step / observation / action 这些基本接口
    2. reward 模块:
       需要 target pose、measured velocity、command 等信息
    3. SIL / discriminator:
       需要 imitation observation 作为模仿子空间
    4. trajectory selector:
       需要 target pose 和 imitation sequence 做 DTW
    """

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """
        返回环境完整观测向量的维度。

        例如可以包含：
        - base 姿态
        - base 速度
        - 关节位置
        - 关节速度
        - 上一时刻动作
        - command 编码
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """
        返回动作向量维度。

        在四足机器人任务里通常对应：
        - 关节目标位置
        - 关节目标速度
        - 或者扭矩 / action residual
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def imitation_obs_dim(self) -> int:
        """
        返回 imitation observation 的维度。

        这部分观测通常是完整 observation 的一个子空间，
        用于：
        - 任务奖励中的姿态跟踪
        - SIL 判别器输入
        - target pose DTW 比较
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def num_skills(self) -> int:
        """
        返回环境中定义的技能数量。
        """
        raise NotImplementedError

    @property
    def command_dim(self) -> int:
        """
        返回离散技能命令的 one-hot 维度。

        默认与技能数相同；如果后续存在更复杂的 command 编码，
        子类可以覆盖这个属性。
        """
        return self.num_skills

    @property
    def velocity_range(self) -> tuple[float, float]:
        """
        返回速度命令采样范围。

        默认值与论文中描述保持一致。
        """
        return (-0.5, 0.5)

    @property
    def skill_names(self) -> list[str]:
        """
        返回所有技能名称。

        基类给出一个通用默认实现：`skill_0`, `skill_1`, ...
        如果子类有更具体的技能语义，建议覆盖。
        """
        return [f"skill_{index}" for index in range(self.num_skills)]

    @abstractmethod
    def reset(
        self,
        command: PasistCommand | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        重置环境。

        参数:
        - command:
          可选的技能命令。如果传入，则环境应围绕该命令初始化当前 episode；
          如果不传入，环境可以自行采样一个默认 command。
        - seed:
          用于控制环境随机性，便于复现实验。

        返回:
        - observation:
          shape = [obs_dim] 的初始观测
        - info:
          附加信息字典。建议至少包含：
          - "command"
          - "skill_id"
          - "target_pose"
          - "imitation_obs"
          - "measured_velocity"

        注意:
        - trainer 不应该假定环境自己计算了 task reward / regularization reward
        - 但如果子类愿意，也可以在 info 里附带这些中间量
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        执行动作并推进环境一个时间步。

        参数:
        - action:
          shape = [action_dim] 的动作向量

        返回:
        - next_observation:
          shape = [obs_dim]
        - env_reward:
          环境原生奖励。对于 PASIST 来说，这个值可以是占位值，
          因为实际训练中通常会重新组合 task / SIL / regularization reward。
        - terminated:
          环境自然终止标志，例如跌倒或达成任务结束
        - truncated:
          时间截断标志，例如超过 episode 最大步数
        - info:
          建议至少包含：
          - "command"
          - "skill_id"
          - "target_pose"
          - "imitation_obs"
          - "measured_velocity"
        """
        raise NotImplementedError

    @abstractmethod
    def sample_command(self, skill_id: int | None = None) -> PasistCommand:
        """
        采样一个新的训练 command。

        参数:
        - skill_id:
          如果传入，则固定采样该技能；
          否则由环境自行随机选择技能。

        返回:
        - PasistCommand
        """
        raise NotImplementedError

    @abstractmethod
    def get_target_pose(self, skill_id: int) -> np.ndarray:
        """
        返回指定 skill 对应的 target pose。

        返回值通常应为 shape = [imitation_obs_dim] 的一维向量，
        这样可以直接用于：
        - `compute_task_reward()`
        - `TrajectorySelector.replicate_target_pose()`
        """
        raise NotImplementedError

    @abstractmethod
    def extract_imitation_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        从完整 observation 中提取 imitation observation。

        参数:
        - observation:
          shape = [obs_dim]

        返回:
        - imitation_observation:
          shape = [imitation_obs_dim]

        这是 PASIST 环境中最关键的辅助接口之一，因为它统一了：
        - task reward 的姿态比较
        - discriminator 输入
        - DTW 轨迹筛选
        """
        raise NotImplementedError

    def build_command(self, velocity: float, skill_id: int) -> PasistCommand:
        """
        根据速度和 skill_id 构造标准化的 PasistCommand。

        这是一个通用辅助函数，具体环境子类通常不需要重写。
        它能减少 `sample_command()` 中重复写 one-hot / name 的样板代码。
        """
        skill_id = int(skill_id)
        if not 0 <= skill_id < self.num_skills:
            raise ValueError(f"skill_id 超出范围: {skill_id}")

        low, high = self.velocity_range
        velocity = float(np.clip(float(velocity), low, high))
        one_hot = np.zeros(self.command_dim, dtype=np.float32)
        one_hot[skill_id] = 1.0
        return PasistCommand(
            velocity=velocity,
            skill_id=skill_id,
            skill_name=self.skill_names[skill_id],
            one_hot=one_hot,
        )

    def build_info(
        self,
        observation: np.ndarray,
        command: PasistCommand,
        measured_velocity: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        构造训练中常用的 info 字典。

        作用:
        - 让不同环境子类的 info 结构尽量一致
        - 减少子类重复拼装 `command / target_pose / imitation_obs` 的代码

        参数:
        - observation:
          当前完整观测
        - command:
          当前生效的技能命令
        - measured_velocity:
          当前环境测得的速度；如果没有可传 None
        - extra:
          额外补充信息，会被 merge 到返回字典中

        返回:
        - 标准化的 info 字典
        """
        info = {
            "command": command,
            "skill_id": int(command.skill_id),
            "skill_name": command.skill_name,
            "target_pose": self.get_target_pose(command.skill_id).copy(),
            "imitation_obs": self.extract_imitation_observation(observation).copy(),
            "measured_velocity": (
                None
                if measured_velocity is None
                else (
                    float(measured_velocity)
                    if np.asarray(measured_velocity).ndim == 0
                    else np.asarray(measured_velocity, dtype=np.float32).copy()
                )
            ),
        }
        if extra:
            info.update(extra)
        return info

    def measure_velocity(self, observation: np.ndarray) -> float:
        """
        从 observation 中估计当前速度。

        基类默认返回 0.0，因为不同环境的速度定义差异很大。
        如果你的 observation 中明确包含 base x 速度等量，建议子类覆盖。
        """
        _ = observation
        return 0.0
