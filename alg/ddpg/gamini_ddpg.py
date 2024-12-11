import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
# 环境
env = gym.make('MountainCarContinuous-v0')


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        return self.l3(q)


class DDPG:
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256, lr=3e-4, gamma=0.99, tau=5e-3,
                 buffer_capacity=1000000):
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        action = (action + np.random.normal(0, noise, size=env.action_space.shape[0]))
        return action.clip(env.action_space.low, env.action_space.high)

    def update(self, batch_size):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).unsqueeze(1).to(device)

        # 计算目标 Q 值
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()

        # 计算当前 Q 值
        current_Q = self.critic(state, action)

        # 计算评论家损失
        critic_loss = F.mse_loss(current_Q, target_Q)

        # 更新评论家网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 计算演员损失
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # 更新演员网络
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# 超参数
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
hidden_dim = 256
lr = 3e-4
gamma = 0.99
tau = 5e-3
buffer_capacity = 1000000
batch_size = 256
episodes = 500
max_steps = 200
render_interval = 25

# 初始化 DDPG 智能体
agent = DDPG(state_dim, action_dim, max_action, hidden_dim, lr, gamma, tau, buffer_capacity)

# 训练循环
rewards = []
for episode in range(episodes):
    state = env.reset()
    episode_reward = 0
    for step in range(max_steps):
        if episode % render_interval == 0:
            env.render()

        action = agent.select_action(state)
        next_state, reward, terminated, _ = env.step(action)
        done = terminated

        # MountainCarContinuous-v0 的奖励需要重新设计
        reward = next_state[0] + 0.5
        if next_state[0] >= 0.5:
            reward += 10

        agent.replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if len(agent.replay_buffer) > batch_size:
            agent.update(batch_size)

        if done:
            break

    rewards.append(episode_reward)
    print(f"Episode: {episode + 1}, Reward: {episode_reward:.2f}")

env.close()

# 绘制奖励曲线
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DDPG on MountainCarContinuous-v0")
plt.show()
