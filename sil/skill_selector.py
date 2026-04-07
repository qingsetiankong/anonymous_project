from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SkillCommand:
    """
    一个训练时发送给策略的 command。

    字段:
    - velocity: 连续速度命令 v
    - skill_id: 离散技能编号
    - one_hot: skill_id 对应的 one-hot 编码
    """

    velocity: float
    skill_id: int
    one_hot: np.ndarray


class SkillSelector:
    """
    PASIST 中的技能选择器。

    核心思想:
    - 记录每个 skill 当前的平均 task reward
    - 用“当前平均奖励 / 该 skill 理想奖励”的比值估计掌握程度
    - 对已经掌握得更好的 skill 降低采样概率
    - 对仍然困难、奖励较低的 skill 提高采样概率

    这样可以缓解 mode collapse，避免策略总是偏向容易学的技能。
    """

    def __init__(
        self,
        num_skills: int,
        optimal_task_rewards: list[float] | np.ndarray | None = None,
        velocity_range: tuple[float, float] = (-0.5, 0.5),
        temperature: float = 1.0,
        epsilon: float = 1e-6,
        random_seed: int = 0,
    ) -> None:
        if num_skills <= 0:
            raise ValueError("num_skills 必须为正整数")

        self.num_skills = int(num_skills)
        self.velocity_range = (float(velocity_range[0]), float(velocity_range[1]))
        self.temperature = float(max(temperature, epsilon))
        self.epsilon = float(epsilon)
        self._rng = np.random.default_rng(random_seed)

        if optimal_task_rewards is None:
            self.optimal_task_rewards = np.ones(self.num_skills, dtype=np.float32)
        else:
            rewards = np.asarray(optimal_task_rewards, dtype=np.float32)
            if rewards.shape != (self.num_skills,):
                raise ValueError("optimal_task_rewards 的长度必须等于 num_skills")
            self.optimal_task_rewards = np.maximum(rewards, self.epsilon)

        # reward_sums / counts 用于计算每个 skill 的经验平均 task reward
        self.reward_sums = np.zeros(self.num_skills, dtype=np.float64)
        self.counts = np.zeros(self.num_skills, dtype=np.int64)

    def one_hot(self, skill_id: int) -> np.ndarray:
        """
        返回指定技能的 one-hot 编码。
        """
        one_hot = np.zeros(self.num_skills, dtype=np.float32)
        one_hot[int(skill_id)] = 1.0
        return one_hot

    def update(self, skill_id: int, task_reward: float, weight: float = 1.0) -> None:
        """
        用一条新样本更新某个 skill 的统计量。

        参数:
        - skill_id: 技能编号
        - task_reward: 对应样本或 episode 的任务奖励
        - weight: 该样本计入统计时的权重
        """
        skill_id = int(skill_id)
        self.reward_sums[skill_id] += float(task_reward) * float(weight)
        self.counts[skill_id] += int(max(round(weight), 1))

    def batch_update(self, skill_ids, task_rewards) -> None:
        """
        批量更新多个 skill 的奖励统计。
        """
        for skill_id, task_reward in zip(skill_ids, task_rewards):
            self.update(skill_id=int(skill_id), task_reward=float(task_reward))

    def average_task_rewards(self) -> np.ndarray:
        """
        返回每个 skill 的经验平均 task reward。

        如果某个 skill 还没有统计数据，则默认平均奖励为 0。
        """
        averages = np.zeros(self.num_skills, dtype=np.float32)
        valid_mask = self.counts > 0
        averages[valid_mask] = (self.reward_sums[valid_mask] / self.counts[valid_mask]).astype(np.float32)
        return averages

    def mastery_ratios(self) -> np.ndarray:
        """
        计算每个 skill 当前的“掌握比例”。

        对应论文中的:
        p(skill) = 平均 task reward / 理想 task reward

        值越大，表示该 skill 越接近目标水平。
        """
        return self.average_task_rewards() / self.optimal_task_rewards

    def sampling_probabilities(self) -> np.ndarray:
        """
        根据当前掌握比例计算各个 skill 的采样概率。

        直觉上:
        - 掌握比例高 -> 说明这个 skill 已经比较会了 -> 少采样
        - 掌握比例低 -> 说明这个 skill 还没学好 -> 多采样

        为了防止某个 skill 概率为 0，这里加入 epsilon 稳定项。
        """
        mastery = self.mastery_ratios()
        deficits = 1.0 / (mastery + self.epsilon)
        logits = deficits / self.temperature
        probabilities = logits / logits.sum()
        return probabilities.astype(np.float32)

    def sample_skill_id(self) -> int:
        """
        按当前采样分布抽样一个 skill id。
        """
        probabilities = self.sampling_probabilities()
        return int(self._rng.choice(self.num_skills, p=probabilities))

    def sample_velocity(self) -> float:
        """
        在给定范围内均匀采样一个速度命令。
        """
        low, high = self.velocity_range
        return float(self._rng.uniform(low, high))

    def sample_command(self, skill_id: int | None = None) -> SkillCommand:
        """
        采样一个完整 command = (velocity, skill_id, one_hot)。

        参数:
        - skill_id: 若传入则固定技能；否则根据 selector 当前策略自动采样
        """
        chosen_skill = self.sample_skill_id() if skill_id is None else int(skill_id)
        velocity = self.sample_velocity()
        return SkillCommand(
            velocity=velocity,
            skill_id=chosen_skill,
            one_hot=self.one_hot(chosen_skill),
        )

    def summary(self) -> dict[str, list[float]]:
        """
        返回选择器当前状态的摘要，便于日志输出。
        """
        return {
            "average_task_rewards": self.average_task_rewards().tolist(),
            "mastery_ratios": self.mastery_ratios().tolist(),
            "sampling_probabilities": self.sampling_probabilities().tolist(),
        }
