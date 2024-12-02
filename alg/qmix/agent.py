import numpy as np
import torch
from alg.qmix.policy import QMIX


class Agents:
    def __init__(self, conf):
        self.conf = conf
        self.device = conf.device
        self.n_actions = conf.n_actions
        self.n_agents = conf.n_agents
        self.state_shape = conf.state_shape
        self.obs_shape = conf.obs_shape
        self.episode_limit = conf.episode_limit

        self.policy = QMIX(conf)

        print("QMIX Agents inited!")

    def choose_action(self, obs, last_action, agent_num, availible_actions, epsilon):
        inputs = obs.copy()
        # print(availible_actions)
        availible_actions_idx = np.nonzero(availible_actions)[0]
        agents_id = np.zeros(self.n_agents)  # [0.  0.  0.  0. ]
        # 该智能体置为1
        agents_id[agent_num] = 1.  # 对于agent_0: [1.  0.  0.  0. ]

        if self.conf.last_action:
            inputs = np.hstack((inputs, last_action))  # 314后面加上两位, 变成316位 【[0 1]表示上次选择了动作2】
        if self.conf.reuse_network:
            inputs = np.hstack((inputs, agents_id))  # 继续在后面加上n_agent位，变成320位 【[1.  0.  0.  0.]表示该智能体是0号智能体】
        hidden_state = self.policy.eval_hidden[:, agent_num, :]
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(self.device)  # (42,) -> (1,42)
        availible_actions = torch.tensor(availible_actions, dtype=torch.float32).unsqueeze(0).to(self.device)

        # get q value
        q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn_net(inputs, hidden_state)
        # choose action form q value
        q_value[availible_actions == 0.0] = -float("inf")
        if np.random.uniform() < epsilon:
            action = np.random.choice(availible_actions_idx)  # 直接得到0或1
        else:
            action_ = torch.argmax(q_value)  # tensor(0)或tensor(1)
            action = action_.item()
        return action

    # def _get_max_episode_len(self, batch):
    #     max_episode_len = 0
    #     for episode_data in batch:
    #         terminated = episode_data["terminated"]  # 挨个取出数据
    #         for transition_idx in range(self.episode_limit):  # 取出每个数据中每一步的done
    #             # print(terminated[0][3][0])
    #             for agent_terminated in terminated[0][transition_idx][0]:  # 挨个检查agent的done
    #                 if agent_terminated:  # 如果有一个done，则返回该done所在的位置
    #                     if transition_idx + 1 >= max_episode_len:
    #                         max_episode_len = transition_idx + 1
    #                     break
    #                 elif transition_idx == self.episode_limit-1:
    #                     max_episode_len = self.episode_limit
    #                     break
    #                 else:
    #                     print("最大步长计算错误，请检查qmix中的agent._get_max_episode_len")
    #     return max_episode_len

    def _get_max_episode_len(self, batch):
        terminated = batch["terminated"]
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.episode_limit):
                if terminated[episode_idx, transition_idx, 0]:
                    if transition_idx+1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break  # 如果这场的done找到了，则就直接下一场就行
                elif transition_idx == self.episode_limit - 1:
                    max_episode_len = self.episode_limit
                    break
        return max_episode_len



    def _get_min_episode_len(self, batch):
        min_episode_len = self.episode_limit
        for episode_data in batch:
            terminated = episode_data["terminated"]  # 挨个取出数据
            for transition_idx in range(self.episode_limit):  # 取出每个数据中每一步的done
                # print(terminated[0][3][0])
                for agent_terminated in terminated[0][transition_idx][0]:  # 挨个检查agent的done
                    if agent_terminated:  # 如果有一个done，则返回该done所在的位置
                        if transition_idx + 1 <= min_episode_len:
                            min_episode_len = transition_idx + 1
                        break
        return min_episode_len

    def train(self, batch, train_step, epsilon=None):
        # 不同的episode的数据长度不同，因此需要得到最大长度，后面会进行补全
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]  # 根据最大长度进行截取
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.conf.save_frequency == 0:
            self.policy.save_model(train_step)
            print("已保存网络")

