import torch.nn as nn
import torch
import torch.nn.functional as F


# ----------------------------------- #
# 构建单个智能体的RNN网络
# ----------------------------------- #
class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, hidden_dim, n_action):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_shape, self.hidden_dim)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)  # GRUCell(input_size, hidden_size)
        self.fc2 = nn.Linear(self.hidden_dim, n_action)

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)  # GRUCell 的输入要求（current_input, last_hidden_state）
        q = self.fc2(h)  # h 是这一时刻的隐状态，用于输到下一时刻的RNN网络中去，q 是真实行为Q值输出
        return q, h


# ----------------------------------- #
# 构建上层的QMIX网络
# ----------------------------------- #
class QMixNet(nn.Module):
    def __init__(self, state_shape, hidden_dim, n_agents):
        super().__init__()

        self.state_shape = state_shape
        self.hidden_dim = hidden_dim
        self.n_agents = n_agents

        # 因为生成的 hyper_w1 需要是一个矩阵，而 pytorch 神经网络只能输出一个向量，
        # 所以就先输出长度为需要的 矩阵行*矩阵列 的向量，然后再转化成矩阵

        # hyper_w1 网络用于输出推理网络中的第一层神经元所需的 weights，
        # 推理网络第一层需要 qmix_hidden * n_agents 个偏差值，因此 hyper_w1 网络输出维度为 qmix_hidden * n_agents
        self.hyper_w1 = nn.Sequential(nn.Linear(state_shape, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, n_agents * hidden_dim))

        # hyper_w2 生成推理网络需要的从隐层到输出 Q 值的所有 weights，共 qmix_hidden 个
        self.hyper_w2 = nn.Sequential(nn.Linear(state_shape, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim))

        # hyper_b1 生成第一层网络对应维度的偏差 bias
        self.hyper_b1 = nn.Linear(state_shape, hidden_dim)
        # hyper_b2 生成对应从隐层到输出 Q 值层的 bias
        self.hyper_b2 = nn.Sequential(nn.Linear(state_shape, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, 1)
                                      )

    def forward(self, q_values, states):  # states的shape为(episode_num, max_episode_len， state_shape)
        # 传入的q_values是三维的，是所有智能体的q值。shape为(episode_num, max_episode_len， n_agents)
        episode_num = q_values.size(0)
        # 在 view 方法中，-1 是一个特殊的占位符，表示该维度的大小由其他维度的大小自动推导而来。具体推导过程如下：
        # 假设 episode_num 为 E，max_episode_len 为 M，n_agents 为 A。
        # 原始形状的总元素数为 E * M * A。
        # 目标形状的总元素数也必须为 E * M * A。
        # 在目标形状 (episode_num * max_episode_len, 1, n_agents) 中，第一个维度的大小为 E * M，第二个维度的大小为 1，第三个维度的大小为 A。
        # 因此，-1 会被自动计算为 E * M，即 episode_num * max_episode_len
        q_values = q_values.view(-1, 1, self.n_agents)  # (episode_num * max_episode_len, 1, n_agents)
        states = states.reshape(-1, self.state_shape)  # (episode_num * max_episode_len, state_shape)

        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)

        w1 = w1.view(-1, self.n_agents, self.hidden_dim)
        b1 = b1.view(-1, 1, self.hidden_dim)
        # 对于每个批量中的样本，q_values 的形状为 (n_agents, n_actions)，w1 的形状为 (n_actions, hidden_dim)。
        # torch.bmm 会将每个 q_values 样本的矩阵与对应的 w1 样本的矩阵相乘，结果的形状为 (n_agents, hidden_dim)
        hidden = F.elu(torch.bmm(q_values, w1) + b1)  # torch.bmm(a, b) 计算矩阵 a 和矩阵 b 相乘

        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)

        w2 = w2.view(-1, self.hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(episode_num, -1, 1)
        return q_total
