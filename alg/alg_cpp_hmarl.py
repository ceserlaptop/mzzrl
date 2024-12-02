import torch
import numpy as np
import networks
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class Alg(object):

    def __init__(self, obs_size, action_size, skill_num, arg):
        """
        Args:
            obs_size: 观测维度
            action_size: 动作维度
            skill_num: 技巧维度
            arg: 输入的参数
        """
        # 三个网络，编码器，上层和下层
        self.ppo_optim = None
        self.qmix_optimizer = None
        self.qmix = None
        self.ppo = None
        self.decoder = None

        # 各个空间的维度
        self.obs_size = obs_size
        self.action_size = action_size
        self.skill_num = skill_num
        self.agent_num = arg.agent_num
        self.arg = arg

        # 网络参数
        # self.n_agents = n_agents
        # self.tau = config_alg['tau']
        # self.lr_Q = config_alg['lr_Q']
        # self.lr_actor = config_alg['lr_actor']
        # self.lr_decoder = config_alg['lr_decoder']
        # self.gamma = config_alg['gamma']
        #
        # 轨迹相关参数
        self.traj = None
        self.decoder_probs = None
        self.decoder_out = None
        self.traj_length = arg.high_net_step
        self.traj_skip = arg.traj_skip
        self.traj_length_downsampled = int(np.ceil(self.traj_length / self.traj_skip))
        self.use_state_difference = arg.use_state_difference
        if self.use_state_difference:
            self.traj_length_downsampled -= 1

        # Domain-specific removal of information from agent observation
        # Either none (deactivate) or scalar index where obs should be truncated for use by decoder
        self.obs_truncate_length = arg.obs_truncate_length
        assert ((self.obs_truncate_length is None) or (self.obs_truncate_length <= self.obs_size))
        #
        self.low_level_alg = arg.low_level_alg
        self.high_level_alg = arg.high_level_alg
        # assert (self.low_level_alg == 'reinforce' or self.low_level_alg == 'iac' or self.low_level_alg == 'iql')
        # if self.low_level_alg == 'iac':
        #     self.lr_V = config_alg['lr_V']

        # Initialize computational graph
        self.create_networks()
        # self.list_initialize_target_ops, self.list_update_target_ops, self.list_update_target_ops_low =
        # self.get_assign_target_ops()
        # self.create_train_op_high()
        # self.create_train_op_low()
        # self.create_train_op_decoder()

        # TF summaries
        # self.create_summary()

    def create_networks(self):

        # 定义编码器网络
        if self.obs_truncate_length:
            # 输入数据举例：表示batch_size为1，轨迹长度为self.traj_length_downsampled，每个时间步观测长度为self.obs_truncate_length
            # 定义为输入数据长度，每个数据的维度，隐藏层神经元数量，输出维度
            self.decoder = networks.Decoder(self.traj_length_downsampled, self.obs_truncate_length,
                                            self.arg.decode_hidden_size, self.skill_num)
        else:
            self.decoder = networks.Decoder(self.traj_length_downsampled, self.obs_size, self.arg.decode_hidden_size,
                                            self.skill_num)

        # 定义下层网络ppo
        if self.low_level_alg == 'ppo':
            # 定义为输入数据长度（self.obs_size+self.skill_num），隐藏层神经元数量，输出维度
            self.ppo = networks.PPO(self.obs_size, self.skill_num, self.arg.n_h1_low, self.arg.n_h2_low,
                                    self.action_size)
            self.ppo_optim = torch.optim.Adam(self.ppo.parameters(), lr=self.arg.lr_ppo)

        # 定义上层网络qmix
        # 定义为输入数据长度（self.obs_size+self.skill_num），智能体数量，隐藏层神经元数量，输出为q_tot
        if self.high_level_alg == 'qmix':
            self.qmix = networks.QMIX(self.obs_size, self.skill_num, self.agent_num, self.arg.n_h1_high,
                                      self.arg.n_h2_high, self.arg.n_h_high_mixer)
            self.qmix_optimizer = optim.Adam(self.qmix.parameters(), lr=self.arg.lr_qmix)

    def compute_reward(self, agents_traj_obs, role):
        """
        Computes P(z|traj) as the reward for a low-level policy.

        Args:
            agents_traj_obs: Tensor of shape (n_agents, traj_length, obs_dim).
            role: Tensor of shape (n_agents, n_roles), one-hot encoded roles.

        Returns:
            prob: Tensor of shape (n_agents,), representing P(z|traj) for each agent.
        """
        # 沿轨迹维度采样，得到采样后的轨迹
        obs_downsampled = agents_traj_obs[:, ::self.traj_skip, :]
        if self.obs_truncate_length:
            obs_downsampled = obs_downsampled[:, :, :self.obs_truncate_length]

        if self.use_state_difference:
            # 使用轨迹中连续状态之间的差异，而不是状态
            obs_downsampled = obs_downsampled[:, 1:, :] - obs_downsampled[:, :-1, :]

        assert (obs_downsampled.shape[1] == self.traj_length_downsampled)

        # Pass through decoder
        _, decoder_probs = self.decoder(obs_downsampled)  # (n_agents, n_roles)

        # Compute P(z|traj)
        prob = torch.sum(decoder_probs * role, dim=1)  # (n_agents,)

        return prob

    def assign_roles(self, obs_list, epsilon, roles_num):
        """
        Get high-level role assignment actions for all agents.

        Args:
            obs_list: List of observation vectors, one per agent (shape: [n_agents, obs_size]).
            epsilon: Exploration parameter.
            roles_num: Number of currently activated role dimensions.

        Returns:
            roles: Numpy array of role indices for all agents (shape: [n_agents]).
        """
        # Convert observations to a torch tensor
        obs = torch.tensor(obs_list, dtype=torch.float32)  # Shape: [n_agents, obs_size]

        # Pass observations through the QMIX network to get Q-values
        q_values = self.qmix(obs)  # Shape: [n_agents, skill_num]

        # Compute argmax roles for each agent
        roles_argmax = torch.argmax(q_values[:, :roles_num], dim=1).cpu().numpy()  # Shape: [n_agents]

        # Initialize role assignments
        roles = np.zeros(self.agent_num, dtype=int)

        # Assign roles using epsilon-greedy policy
        for idx in range(self.agent_num):
            if np.random.rand() < epsilon:
                roles[idx] = np.random.randint(0, roles_num)
            else:
                roles[idx] = roles_argmax[idx]

        return roles

    def train_policy_high(self, batch):
        """Training step for high-level policy."""
        # Process the batch data
        obs, roles_1hot, reward, obs_next, done = self.process_batch_high(batch)

        # Convert the input data to PyTorch tensors
        obs = torch.tensor(obs, dtype=torch.float32)
        obs_next = torch.tensor(obs_next, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        roles_1hot = torch.tensor(roles_1hot, dtype=torch.float32)

        # 计算当前的Q值
        q_tot = self.qmix(obs, roles_1hot)  # (batch_size, 1)

        # 计算目标Q值
        with torch.no_grad():
            # 计算下一个状态的Q值
            next_q_tot = self.qmix(obs_next, roles_1hot)  # (batch_size, 1)
            target_q = reward + (1 - done) * self.arg.gamma * next_q_tot

        # 计算损失
        loss = F.mse_loss(q_tot.squeeze(), target_q.squeeze())  # 使用均方误差损失

        # 更新网络参数
        self.qmix_optimizer.zero_grad()
        loss.backward()
        self.qmix_optimizer.step()

    def process_batch_high(self, batch):
        """
        Processes the batch data and prepares it for training.
        This function should extract states, actions, rewards, etc.

        Args:
            batch: Raw batch of data.
        """
        # Example processing (adjust according to your actual batch format)
        obs = np.stack(batch[:, 0])  # [batch*n_agents, obs_dim]
        roles_int = np.stack(batch[:, 1])  # [batch*n_agents, l_z]
        reward = np.stack(batch[:, 2])  # [batch*n_agents]
        obs_next = np.stack(batch[:, 3])  # [batch*n_agents, obs_dim]
        done = np.stack(batch[:, 4])  # [batch*n_agents]

        # Try to free memory
        batch = None
        n_steps = obs.shape[0]

        # In-place reshape for obs, so that one time step
        # for one agent is considered one batch entry
        obs.shape = (n_steps * self.agent_num, self.obs_size)
        obs_next.shape = (n_steps * self.agent_num, self.obs_size)
        roles_1hot = self.process_actions(n_steps, roles_int, self.skill_num)

        return obs, roles_1hot, reward, obs_next, done

    def process_batch_low(self, batch):
        """
        Processes the batch data and prepares it for training.
        This function should extract states, actions, rewards, etc.

        Args:
            batch: Raw batch of data.
        """
        # Example processing (adjust according to your actual batch format)
        obs = np.stack(batch[:, 0])  # [batch*n_agents, obs_dim]
        action_int = np.stack(batch[:, 1])  # [batch*n_agents, l_z]
        local_rewards = np.stack(batch[:, 2])  # [batch*n_agents]
        obs_next = np.stack(batch[:, 3])  # [batch*n_agents, obs_dim]
        roles = np.stack(batch[:, 4])
        done = np.stack(batch[:, 5])  # [batch*n_agents]

        # Try to free memory
        batch = None
        n_steps = obs.shape[0]

        # In-place reshape for obs, so that one time step
        # for one agent is considered one batch entry
        obs.shape = (n_steps * self.agent_num, self.obs_size)
        obs_next.shape = (n_steps * self.agent_num, self.obs_size)
        local_rewards.shape = (n_steps * self.agent_num)
        roles.shape = (n_steps * self.agent_num, self.skill_num)
        done = np.repeat(done, self.agent_num, axis=0)
        actions_1hot = self.process_actions(n_steps, action_int, self.action_size)

        return obs, actions_1hot, local_rewards, obs_next, roles, done

    def process_actions(self, n_steps, actions, n_actions):
        """
        Args:
            n_steps: number of steps in trajectory
            actions: must have shape [time, n_agents], and values are action indices
            n_actions: dimension of action space

        Returns: 1-hot representation of actions
        """
        # Each row of actions is one time step,
        # row contains action indices for all agents
        # Convert to [time, agents, N_roles]
        # so each agent gets its own 1-hot row vector
        actions_1hot = np.zeros([n_steps, self.agent_num, n_actions], dtype=int)
        grid = np.indices((n_steps, self.agent_num))
        actions_1hot[grid[0], grid[1], actions] = 1

        # In-place reshape of actions to [time*n_agents, N_roles]
        actions_1hot.shape = (n_steps * self.agent_num, n_actions)

        return actions_1hot

    def choose_actions(self, obs_list, roles, epsilon):
        if self.low_level_alg == 'ppo':
            logits, value = self.ppo(obs_list, roles)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()  # 采样生成动作
            # log_prob = dist.log_prob(action)
            return action

    def train_policy_low(self, batch, epsilon_clip=0.2, value_loss_coef=0.5, entropy_coef=0.01,):
        obs, actions_1hot, local_rewards, obs_next, roles, done = self.process_batch_low(batch)

        obs = torch.tensor(obs, dtype=torch.float32)
        actions_1hot = torch.tensor(actions_1hot, dtype=torch.float32)
        local_rewards = torch.tensor(local_rewards, dtype=torch.float32)
        obs_next = torch.tensor(obs_next, dtype=torch.float32)
        roles = torch.tensor(roles, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        with torch.no_grad():
            _, next_values = self.ppo(obs_next, roles)
            next_values = next_values.squeeze(-1)

        # 计算旧策略的 log 概率和价值
        logits, values = self.ppo(obs, roles)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        old_log_probs = dist.log_prob(actions_1hot.argmax(dim=-1))  # 动作需要为索引形式
        values = values.squeeze(-1)

        # 计算当前策略的 log 概率和价值
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions_1hot.argmax(dim=-1))
        entropy = dist.entropy()
        current_values = values.squeeze(-1)

        # 计算优势函数（使用折扣回报减去旧的状态值）
        advantages = local_rewards - values.detach()

        # 策略损失
        ratios = torch.exp(log_probs - old_log_probs.detach())
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - epsilon_clip, 1 + epsilon_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 价值损失
        value_loss = F.mse_loss(current_values, local_rewards)

        # 总损失
        loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy.mean()

        # 更新网络
        self.ppo_optim.zero_grad()
        loss.backward()
        self.ppo_optim.step()

    def train_decoder(self, dataset):
        """Training step for skill decoder.

        Args:
            dataset: list of np.array objects

        """
        # 转换为 PyTorch 张量
        dataset = np.array(dataset)
        obs = np.stack(dataset[:, 0])  # [batch, traj, l_obs]
        z = np.stack(dataset[:, 1])  # [batch, N_roles]

        # Downsample obs along the traj dimension
        # obs has shape [batch, traj, l_obs]
        obs_downsampled = obs[:, ::self.traj_skip, :]
        if self.obs_truncate_length:
            obs_downsampled = obs_downsampled[:, :, :self.obs_truncate_length]
        if self.use_state_difference:
            # use the difference between consecutive states in a trajectory, rather than the state
            obs_downsampled = obs_downsampled[:, 1:, :] - obs_downsampled[:, :-1, :]

        # 检查 shape 是否符合要求
        assert (obs_downsampled.shape[1] == self.traj_length_downsampled)

        # 转换为 PyTorch 张量
        obs_downsampled_tensor = torch.tensor(obs_downsampled, dtype=torch.float32)
        z_tensor = torch.tensor(z, dtype=torch.float32)

        # 前向传递
        self.decoder.train()  # 设置为训练模式
        logits, probs = self.decoder(obs_downsampled_tensor)

        # decoder_probs has shape [batch, N_roles]
        # 计算每个样本的概率并对 z 进行加权
        prob = torch.sum(probs * z_tensor, dim=1)
        expected_prob = prob.mean()

        return expected_prob
