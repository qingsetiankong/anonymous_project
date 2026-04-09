import yaml
import gymnasium as gym 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from rl import rl_utils

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, output_activation=None):
        super(MLP, self).__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        if hidden_dims is None:
            hidden_dims = []
        else:
            # 兼容 tuple / 其它可迭代类型，避免 `[input_dim] + hidden_dims`
            # 在 `hidden_dims` 不是 list 时触发类型错误。
            hidden_dims = list(hidden_dims)
        
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        
        self.layers = nn.ModuleList([
            nn.Linear(layer_dims[i], layer_dims[i + 1])
            for i in range(len(layer_dims) - 1)
        ])
        self.output_activation = output_activation

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        if self.output_activation == 'softmax':
            x = F.softmax(x, dim=1)
        return x

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim):
        super(PolicyNet, self).__init__()
        self.model = MLP(input_dim=state_dim, hidden_dims=hidden_dims, output_dim=action_dim, output_activation='softmax')

    def forward(self, x):
        return self.model(x)

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dims):
        super(ValueNet, self).__init__()
        self.model = MLP(input_dim=state_dim, hidden_dims=hidden_dims, output_dim=1, output_activation=None)

    def forward(self, x):
        return self.model(x)

class ActorCritic:
    def __init__(self, state_dim, action_dim, actor_hidden_dims, critic_hidden_dims, actor_lr, critic_lr, gamma, device):
        self.device = device
        self.gamma = gamma
        self.actor = PolicyNet(state_dim, actor_hidden_dims, action_dim).to(device)
        self.critic = ValueNet(state_dim, critic_hidden_dims).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float32).to(self.device)
        probs = self.actor(state)
        # Gymnasium 返回的是 numpy 数组，需要转 tensor
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        # 确保数据类型正确
        states = torch.tensor(transition_dict['states'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32).to(self.device)
        # Gymnasium 的 done 是 bool，需要转 float
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).view(-1, 1).to(self.device)

        # TD target
        # 注意：这里乘以 (1 - dones) 是为了在 episode 结束时截断
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)

        # actor loss
        probs = self.actor(states)
        # 增加一个小常数防止 log(0)
        log_probs = torch.log(probs.gather(1, actions) + 1e-8)
        actor_loss = torch.mean(-log_probs * td_delta.detach())

        # critic loss
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    required_keys = ["env", "train", "model"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"配置缺少 {key}")
    return config

def test_train():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "training", "ac.yaml")
    config = load_config(config_path)

    env_name = config["env"]["name"]
    
    if env_name == "CartPole-v1":
        env = gym.make(env_name)
    else:
        env = gym.make(env_name)

    gamma = config["train"]["gamma"]
    actor_lr = config["train"]["actor_lr"]
    critic_lr = config["train"]["critic_lr"]
    gpus = config["train"]["gpus"]
    actor_hidden_dims = config["model"]["actor_layers"]
    critic_hidden_dims = config["model"]["critic_layers"]
    num_episodes = config["train"]["num_episodes"]
    seed = config["train"].get("seed", 0)

    # -1 == cpu
    if gpus != -1 and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f'Using device: {device}')
    print(f'Using seed: {seed}')

    env.reset(seed=seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        device=device
    )

    print("Actor 网络结构：")
    print(agent.actor)
    print("\nCritic 网络结构：")
    print(agent.critic)
    
    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns') 
    plt.title('Actor-Critic on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Actor-Critic on {}'.format(env_name))
    plt.show()

if __name__ == "__main__":
    test_train()
