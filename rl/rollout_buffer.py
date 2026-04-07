import torch


class RolloutBuffer:
    """
    用于 PPO / PASIST 训练过程中的轨迹缓存。

    这个 buffer 以 [time, env, feature] 的形式存储一个 rollout 窗口内的所有数据，
    既服务于 PPO 的 return / advantage 计算，也服务于 PASIST 后续的：
    1. 轨迹提取与筛选
    2. SIL 判别器训练
    3. skill / command 维度的统计分析

    约定：
    - num_steps: 一个 rollout 中沿时间维度采集多少步
    - num_envs: 并行环境个数
    - self.step: 当前已经写入了多少个时间步
    """

    def __init__(
        self,
        num_steps,
        num_envs,
        obs_dim,
        act_dim,
        device,
        imitation_obs_dim=None,
        command_dim=None,
        gamma=0.99,
        gae_lambda=0.95,
    ):
        # rollout 的最大时间长度，例如 PPO 中一次采样 24 / 128 / 512 步
        self.num_steps = num_steps
        # 并行环境数量；Isaac Gym / vectorized env 中通常 > 1
        self.num_envs = num_envs
        # 数据所在设备，通常是 cpu 或 cuda
        self.device = device
        # return / advantage 计算使用的折扣因子
        self.gamma = gamma
        # GAE(lambda) 中的 lambda 系数
        self.gae_lambda = gae_lambda
        # command 的 one-hot 维度；如果不做 command 条件训练可以为 None
        self.command_dim = command_dim

        # 当前时刻的观测，shape = [num_steps, num_envs, obs_dim]
        self.obs = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        # 执行动作后的下一时刻观测，shape = [num_steps, num_envs, obs_dim]
        self.next_obs = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        # 动作，连续动作时通常是实值向量，shape = [num_steps, num_envs, act_dim]
        self.actions = torch.zeros(num_steps, num_envs, act_dim, device=device)
        # 行为策略输出该动作时的 log prob，PPO 需要
        self.log_probs = torch.zeros(num_steps, num_envs, 1, device=device)
        # critic 给出的状态价值 V(s)
        self.values = torch.zeros(num_steps, num_envs, 1, device=device)

        # 总奖励：真正用于 PPO return 计算的奖励
        self.reward_total = torch.zeros(num_steps, num_envs, 1, device=device)
        # 任务奖励 r_T，对应论文中的 task reward
        self.reward_task = torch.zeros(num_steps, num_envs, 1, device=device)
        # 正则奖励 r_R，例如动作平滑、扭矩惩罚等
        self.reward_reg = torch.zeros(num_steps, num_envs, 1, device=device)
        # SIL 奖励 r_SIL，由判别器或自模仿模块给出
        self.reward_sil = torch.zeros(num_steps, num_envs, 1, device=device)

        # 是否结束当前 episode；通常 done = terminated or truncated
        self.dones = torch.zeros(num_steps, num_envs, 1, device=device)
        # 环境自然终止，例如跌倒、成功完成任务
        self.terminateds = torch.zeros(num_steps, num_envs, 1, device=device)
        # 截断终止，例如时间上限触发
        self.truncateds = torch.zeros(num_steps, num_envs, 1, device=device)
        # 当前样本对应的 skill id，便于多技能训练统计
        self.skill_ids = torch.zeros(num_steps, num_envs, 1, dtype=torch.long, device=device)
        # episode 编号；默认 -1 表示调用 add 时未提供
        self.episode_ids = torch.full((num_steps, num_envs, 1), -1, dtype=torch.long, device=device)
        # command 中的速度分量 v
        self.command_speeds = torch.zeros(num_steps, num_envs, 1, device=device)

        # imitation observation，一般是用于 DTW / 判别器的模仿特征
        # 比如关节位置、关键姿态子空间等
        if imitation_obs_dim is not None:
            self.imitation_obs = torch.zeros(num_steps, num_envs, imitation_obs_dim, device=device)
        else:
            self.imitation_obs = None

        # command 中的技能 one-hot 分量 m
        if command_dim is not None:
            self.command_onehots = torch.zeros(num_steps, num_envs, command_dim, device=device)
        else:
            self.command_onehots = None

        # PPO 中的 Monte-Carlo / GAE 回报
        self.returns = torch.zeros(num_steps, num_envs, 1, device=device)
        # PPO 中的 advantage
        self.advantages = torch.zeros(num_steps, num_envs, 1, device=device)
        # 当前已经写入的时间步数，范围 [0, num_steps]
        self.step = 0

    def _valid_steps(self):
        """
        返回当前 buffer 内真正有效的时间步数。

        由于一个 rollout 可能没有写满整个 buffer，
        所以后续读取时必须只读取 [0, self.step) 这部分有效区域。
        """
        return self.step

    def _flatten_valid(self, tensor):
        """
        将 [time, env, feature] 的张量展平成 [time * env, feature]，
        并且只保留已经写入的有效时间区间。

        参数:
        - tensor: 任意一个以 [num_steps, num_envs, ...] 排布的缓存张量

        返回:
        - shape = [valid_steps * num_envs, feature_dim] 的二维张量
        """
        valid_steps = self._valid_steps()
        if valid_steps == 0:
            return tensor[:0].reshape(0, tensor.shape[-1])
        return tensor[:valid_steps].reshape(valid_steps * self.num_envs, -1)

    def _empty_episode(self):
        """
        构造一个空 episode 容器。

        extract_episodes() 会按环境逐步积累数据，当遇到 done 时把当前容器推入列表。
        单独写成函数的好处是：
        - 初始化字段更集中
        - reset 当前 episode 时不容易漏字段
        """
        episode = {
            "obs": [],
            "next_obs": [],
            "actions": [],
            "reward_task": [],
            "reward_reg": [],
            "reward_sil": [],
            "reward_total": [],
            "skill_ids": [],
            "episode_ids": [],
            "command_speeds": [],
            "dones": [],
            "terminateds": [],
            "truncateds": [],
        }
        if self.imitation_obs is not None:
            episode["imitation_obs"] = []
        if self.command_onehots is not None:
            episode["command_onehots"] = []
        return episode

    def add(
        self,
        obs,
        action,
        log_prob,
        value,
        reward_total,
        reward_task,
        reward_reg,
        reward_sil,
        done,
        next_obs,
        skill_id,
        imitation_obs=None,
        episode_id=None,
        terminated=None,
        truncated=None,
        command_speed=None,
        command_onehot=None,
    ):
        """
        向 buffer 写入单个时间步的并行环境数据。

        这里的“单个时间步”指：
        - 同一个 t 上
        - 所有并行环境 env_0 ... env_{N-1}
        的一批样本一起写入

        典型输入张量形状：
        - obs: [num_envs, obs_dim]
        - action: [num_envs, act_dim]
        - log_prob/value/reward_xxx/done: [num_envs, 1]
        - imitation_obs: [num_envs, imitation_obs_dim]
        - command_onehot: [num_envs, command_dim]
        """
        assert self.step < self.num_steps, "RolloutBuffer overflow"

        # 基础 PPO 状态-动作信息
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.values[self.step] = value

        # 奖励分解项
        self.reward_total[self.step] = reward_total
        self.reward_task[self.step] = reward_task
        self.reward_reg[self.step] = reward_reg
        self.reward_sil[self.step] = reward_sil

        # 终止标志与下一状态
        self.dones[self.step] = done
        # 如果外部没有细分 terminated / truncated，则默认把 done 当作 terminated
        self.terminateds[self.step] = done if terminated is None else terminated
        # 如果外部没有提供 truncated，则默认全 0
        self.truncateds[self.step] = torch.zeros_like(done) if truncated is None else truncated
        self.next_obs[self.step] = next_obs

        # 技能条件信息
        self.skill_ids[self.step] = skill_id

        # episode id 用于更稳健地追踪完整轨迹
        if episode_id is not None:
            self.episode_ids[self.step] = episode_id

        # command 速度分量
        if command_speed is not None:
            self.command_speeds[self.step] = command_speed

        # 判别器 / DTW / imitation learning 使用的观测子空间
        if self.imitation_obs is not None and imitation_obs is not None:
            self.imitation_obs[self.step] = imitation_obs

        # command 的 one-hot 技能分量
        if self.command_onehots is not None and command_onehot is not None:
            self.command_onehots[self.step] = command_onehot

        # 时间指针后移一位
        self.step += 1

    def compute_returns_and_advantages(self, last_value):
        """
        使用 GAE(lambda) 计算 returns 和 advantages。

        参数:
        - last_value: rollout 最后一个 next state 的 V(s)，shape = [num_envs, 1]

        说明:
        - 只对已经写入的有效时间步进行计算
        - 使用 self.reward_total 作为 PPO 的训练奖励
        - 结果会原地写入 self.returns 和 self.advantages
        """
        valid_steps = self._valid_steps()
        if valid_steps == 0:
            return

        # advantage 沿时间反向递推；每个环境各维护一份
        advantage = torch.zeros(self.num_envs, 1, device=self.device)
        for t in reversed(range(valid_steps)):
            # rollout 最后一步需要使用外部传入的 bootstrap value
            next_value = last_value if t == valid_steps - 1 else self.values[t + 1]
            # 如果 done=1，则终止 bootstrap
            not_done = 1.0 - self.dones[t]
            # TD residual
            delta = self.reward_total[t] + self.gamma * next_value * not_done - self.values[t]
            # GAE 递推公式
            advantage = delta + self.gamma * self.gae_lambda * not_done * advantage
            self.advantages[t] = advantage
            self.returns[t] = self.advantages[t] + self.values[t]

        # PPO 常见做法：对 advantage 做标准化，提升优化稳定性
        adv = self.advantages[:valid_steps]
        adv_mean = adv.mean()
        adv_std = adv.std() + 1e-8
        self.advantages[:valid_steps] = (adv - adv_mean) / adv_std

    def get_minibatches(self, batch_size, shuffle=True):
        """
        将有效 rollout 数据展平并切分成 mini-batch。

        参数:
        - batch_size: 每个 mini-batch 的样本数
        - shuffle: 是否在样本维度上打乱

        返回:
        - 一个 generator；每次 yield 一个 dict

        这个接口不仅服务 PPO，也为 PASIST 的判别器训练保留了
        imitation_obs / reward_sil / command 等字段。
        """
        valid_steps = self._valid_steps()
        total_samples = valid_steps * self.num_envs
        if total_samples == 0:
            return

        # 先把所有有效数据展平到 [N, feature] 形式，便于统一切 batch
        batch = {
            "obs": self._flatten_valid(self.obs),
            "next_obs": self._flatten_valid(self.next_obs),
            "actions": self._flatten_valid(self.actions),
            "log_probs": self._flatten_valid(self.log_probs),
            "values": self._flatten_valid(self.values),
            "returns": self._flatten_valid(self.returns),
            "advantages": self._flatten_valid(self.advantages),
            "reward_total": self._flatten_valid(self.reward_total),
            "reward_task": self._flatten_valid(self.reward_task),
            "reward_reg": self._flatten_valid(self.reward_reg),
            "reward_sil": self._flatten_valid(self.reward_sil),
            "dones": self._flatten_valid(self.dones),
            "terminateds": self._flatten_valid(self.terminateds),
            "truncateds": self._flatten_valid(self.truncateds),
            "skill_ids": self._flatten_valid(self.skill_ids),
            "episode_ids": self._flatten_valid(self.episode_ids),
            "command_speeds": self._flatten_valid(self.command_speeds),
        }

        if self.imitation_obs is not None:
            batch["imitation_obs"] = self._flatten_valid(self.imitation_obs)
        if self.command_onehots is not None:
            batch["command_onehots"] = self._flatten_valid(self.command_onehots)

        # 采样样本索引；PPO 通常需要随机打乱
        indices = (
            torch.randperm(total_samples, device=self.device)
            if shuffle
            else torch.arange(total_samples, device=self.device)
        )

        # 按 batch_size 逐段返回
        for start in range(0, total_samples, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield {key: value[batch_idx] for key, value in batch.items()}

    def extract_episodes(self):
        """
        从 rollout buffer 中按环境维度提取 episode 序列。

        返回:
        - episodes: list[dict]

        每个 episode dict 内部是“按时间顺序排列的张量列表”，方便后续：
        - 计算整段轨迹的累计 task reward
        - 提取 imitation_obs 序列做 DTW
        - 按 skill / command 筛选高质量轨迹写入 SIL buffer

        设计细节:
        - 只遍历有效时间步
        - 遇到 done 时切分一个 episode
        - rollout 末尾若还有未 done 的残余片段，也会保留下来
        """
        valid_steps = self._valid_steps()
        episodes = []
        # 对每个并行环境分别还原时间序列
        for env_id in range(self.num_envs):
            current = self._empty_episode()
            for t in range(valid_steps):
                # 把该环境在当前时间步的所有关键信息写入当前 episode
                current["obs"].append(self.obs[t, env_id].clone())
                current["next_obs"].append(self.next_obs[t, env_id].clone())
                current["actions"].append(self.actions[t, env_id].clone())
                current["reward_task"].append(self.reward_task[t, env_id].clone())
                current["reward_reg"].append(self.reward_reg[t, env_id].clone())
                current["reward_sil"].append(self.reward_sil[t, env_id].clone())
                current["reward_total"].append(self.reward_total[t, env_id].clone())
                current["skill_ids"].append(self.skill_ids[t, env_id].clone())
                current["episode_ids"].append(self.episode_ids[t, env_id].clone())
                current["command_speeds"].append(self.command_speeds[t, env_id].clone())
                current["dones"].append(self.dones[t, env_id].clone())
                current["terminateds"].append(self.terminateds[t, env_id].clone())
                current["truncateds"].append(self.truncateds[t, env_id].clone())

                if self.imitation_obs is not None:
                    current["imitation_obs"].append(self.imitation_obs[t, env_id].clone())
                if self.command_onehots is not None:
                    current["command_onehots"].append(self.command_onehots[t, env_id].clone())

                # done 表示当前 episode 完整结束，立即切分
                if self.dones[t, env_id].item() > 0.5:
                    episodes.append(current)
                    current = self._empty_episode()

            # rollout 提前结束时，最后一段即使还没 done 也保留，避免轨迹丢失
            if current["obs"]:
                episodes.append(current)
        return episodes

    def reset(self):
        """
        重置写入指针。

        注意：
        - 这里只把 self.step 置零
        - 不会逐元素清空所有缓存张量
        - 旧数据是否安全，取决于读取端是否严格只读 [0, self.step)
        """
        self.step = 0

    def clear(self):
        """
        reset() 的语义别名。

        保留这个接口是为了兼容不同训练代码的命名习惯。
        """
        self.reset()
