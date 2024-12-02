import torch.nn as nn
import torch
import torch.nn.functional as F


# 当个MIXer的网络
class DRQN(nn.Module):
    def __init__(self, input_shape, conf):
        super(DRQN, self).__init__()
        self.conf = conf
        self.fc1 = nn.Linear(input_shape, conf.drqn_hidden_dim)
        self.rnn = nn.GRUCell(conf.drqn_hidden_dim, conf.drqn_hidden_dim)
        self.fc2 = nn.Linear(conf.drqn_hidden_dim, conf.n_actions)

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.conf.drqn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


# QMIX网络
class QMIX_net(nn.Module):
    def __init__(self, conf):
        super(QMIX_net, self).__init__()
        """
        生成的hyper_w1需要是一个矩阵，但是torch NN的输出只能是向量；
        因此先生成一个（行*列）的向量，再reshape
        """
        # print(conf.state_shape)
        self.conf = conf
        if self.conf.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(nn.Linear(self.conf.state_shape, self.conf.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.conf.hyper_hidden_dim,
                                                    self.conf.n_agents * self.conf.qmix_hidden_dim))
            self.hyper_w2 = nn.Sequential(nn.Linear(self.conf.state_shape, self.conf.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.conf.hyper_hidden_dim, self.conf.qmix_hidden_dim))
        else:
            self.hyper_w1 = nn.Linear(self.conf.state_shape, self.conf.n_agents * self.conf.qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(self.conf.state_shape, self.conf.qmix_hidden_dim * 1)

        self.hyper_b1 = nn.Linear(self.conf.state_shape, self.conf.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(self.conf.state_shape, self.conf.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.conf.qmix_hidden_dim, 1))

    # input: (batch_size, n_agents, qmix_hidden_dim)
    # q_values: (episode_num, max_episode_len, n_agents)
    # states shape: (episode_num, max_episode_len, state_shape)
    def forward(self, q_values, states):
        # print(self.conf.state_shape)
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.conf.n_agents)
        states = states.reshape(-1, self.conf.state_shape)

        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        w1 = w1.view(-1, self.conf.n_agents, self.conf.qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.conf.qmix_hidden_dim)

        hidden = F.elu(torch.bmm(q_values, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)
        w2 = w2.view(-1, self.conf.qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(episode_num, -1, 1)

        return q_total


# 编码器的RNN网络
class RNN(nn.Module):
    # 所有 Agent 共享同一网络, 因此 input_shape = obs_shape + n_actions + n_agents（one_hot_code）
    def __init__(self, input_shape, args, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)     # GRUCell(input_size, hidden_size)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, action_dim)

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)                # GRUCell 的输入要求（current_input, last_hidden_state）
        q = self.fc2(h)                      # h 是这一时刻的隐状态，用于输到下一时刻的RNN网络中去，q 是真实行为Q值输出
        return q, h


# ppo的策略网络
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


# ppo的价值网络
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
