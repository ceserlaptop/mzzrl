import os
import numpy as np
from env.cpp_env.arguments_env_test import parse_args
import env.cpp_env.scenarios as scenarios
from env.cpp_env.environment import MultiAgentEnv
import torch
from agent import Agents
from utils import ReplayBuffer
from config import Config

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

conf = Config()
device = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')


def make_env(arglist):
    """
    create the environment from script
    """
    scenario = scenarios.load(arglist.scenario_name + ".py").Scenario(r_cover=0.1, r_comm=0.4, comm_r_scale=0.9,
                                                                      comm_force_scale=5.0, env_size=10,
                                                                      arglist=arglist)
    world = scenario.make_world()
    if arglist.benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            done_callback=scenario.done)
        # env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def train():
    # ----------------------------------- #
    """step1:"""
    # 配置环境
    # ----------------------------------- #
    arg = parse_args()
    env = make_env(arg)

    print('\n=============================')
    print('=1 coverage_Env is right ...')
    print('=============================')

    # ----------------------------------- #
    """step2:"""
    # 配置所有智能体
    # ----------------------------------- #
    agents = Agents(conf)
    # rollout_worker = RolloutWorker(env, agents, conf)
    buffer = ReplayBuffer(conf.buffer_size)
    print('=============================')
    print('=2 The {} agents are inited ...'.format(env.n))
    print('=============================')
    # ----------------------------------- #
    # 模型保存地址
    # ----------------------------------- #
    # save_path = conf.result_dir + conf.map_name
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    # ----------------------------------- #
    """step3:"""
    # 初始化训练参数
    # ----------------------------------- #
    cover_rates = []
    episode_rewards = []
    train_steps = 0
    # memory = ReplayBuffer(conf)  # 初始化buffer
    avail_action = conf.avail_action
    epsilon = conf.epsilon_start
    epsilon_step = conf.epsilon_step

    # ------------------------------------ #
    # 开始训练场数的循环
    # ------------------------------------ #
    for idx_episode in range(conf.max_episode):

        # 初始化游戏开始的参数
        episode_reward = 0

        last_action = np.zeros((conf.n_agents, conf.n_actions))  # [0 0] 初始化上一步的动作
        agents.policy.init_hidden(1)  # 初始化隐藏层状态，（1）表示是第一场
        obs_n = env.reset()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []  # 记录轨迹
        # 开始一场游戏的循环
        idx_step = 0
        state = False
        while idx_step < conf.episode_limit:
            state = np.concatenate(obs_n)
            actions, avail_actions, actions_onehot = [], [], []  # 初始化动作编码
            # 选择各个智能体的动作
            for agent_id in range(conf.n_agents):
                # 输入该智能体的观测，上一个动作，智能体id，可选动作【[1, 1]有几个动作就代表几位（1hot）】，贪婪系数；得到动作0或者1
                action = agents.choose_action(obs_n[agent_id], last_action[agent_id], agent_id, avail_action,
                                              epsilon)
                # 生成动作的onehot编码
                action_onehot = np.zeros(conf.n_actions)  # [0 0]
                action_onehot[action] = 1  # 表示选择动作1 【[0 1]】
                actions.append(action)  # 记录动作【[1, 0, ....]】
                actions_onehot.append(action_onehot)  # 记录动作1hot【[[0 1], [1 0]], ....】
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot  # 把该智能体上次的动作记录下来

            # actions = [np.array([13719068,0.02047865,0.7962253,0.03812775,0.00797753]),
            #            np.array([0.07508944, 0.7004558, 0.04798195, 0.01469667, 0.16177621]),
            #            np.array([0.0214441, 0.748268, 0.10931759, 0.01024642, 0.11072386]),
            #            np.array([0.03743383, 0.2150916, 0.6732575, 0.04681827, 0.02739876]),]

            # 和环境交互
            obs_n_t, rew_n, done_n = env.step(actions)  # 分别表示每个智能体用那个动作【[1, 0, 1, 1]】
            state_t = np.concatenate(obs_n_t)
            obs_n = obs_n_t
            done = False
            if True in done_n:
                done = True
            # env.render()
            # 记录数据
            o.append(obs_n_t)  # 观测
            s.append(state_t)  # todo 环境的状态，不区分智能体，因为状态只有一个，这里后面要换成真实的环境状态
            # 每个时间步各个智能体的动作【[[[1], [0], [1], [1]], [[1], [0], [1], [1]], .......]】
            u.append(np.reshape(actions, [conf.n_agents, 1]))
            u_onehot.append(
                actions_onehot)  # 变成热编码 【[[[0, 1], [1 ,0], [0, 1], [0, 1]], [[0, 1], [1 ,0], [0, 1], [0, 1]], .......]】
            avail_u.append(avail_actions)
            r.append([sum(rew_n)])  # 计算四个智能体的总奖励,以列表格式加入
            terminate.append([done])  # 四个智能体总体的done
            padded.append([0.])
            episode_reward += sum(rew_n)  # 累加总奖励
            idx_step = idx_step + 1
            if True in done_n:
                break

        if idx_episode >= conf.pretrain_episodes and epsilon > conf.epsilon_end:
            epsilon -= epsilon_step

        # 最后一个动作
        o.append(obs_n)
        s.append(state)
        o_ = o[1:]
        s_ = s[1:]
        o = o[:-1]
        s = s[:-1]

        # target q 在last obs需要avail_action
        avail_actions = []
        for agent_id in range(conf.n_agents):
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_ = avail_u[1:]
        avail_u = avail_u[:-1]

        # 当step<self.episode_limit时，输入数据加padding
        for i in range(idx_step, conf.episode_limit):
            o.append(np.zeros((conf.n_agents, conf.obs_shape)))
            u.append(np.zeros([conf.n_agents, 1]))
            s.append(np.zeros(conf.state_shape))
            r.append([0.])
            o_.append(np.zeros((conf.n_agents, conf.obs_shape)))
            s_.append(np.zeros(conf.state_shape))
            u_onehot.append(np.zeros((conf.n_agents, conf.n_actions)))
            avail_u.append(np.zeros((conf.n_agents, conf.n_actions)))
            avail_u_.append(np.zeros((conf.n_agents, conf.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(
            o=o.copy(),
            s=s.copy(),
            u=u.copy(),
            r=r.copy(),
            o_=o_.copy(),
            s_=s_.copy(),
            avail_u=avail_u.copy(),
            avail_u_=avail_u_.copy(),
            u_onehot=u_onehot.copy(),
            padded=padded.copy(),
            terminated=terminate.copy()
        )

        # 记录数据
        for key in episode.keys():
            episode[key] = np.array([episode[key]])  # 把列表全变成数组

        buffer.add(episode)
        print("当前已运行", idx_episode, "场")
        if idx_episode >= conf.pretrain_episodes and idx_episode % conf.train_frequency == 0:  # 每10场训练一次
            for train_step in range(conf.train_steps):  # 一次训练多少轮
                # 一个mini_batch有batch_size个数据每个数据中有11类数据，【其中obs:【一场游戏，20个时间步，四个智能体，314位观测(1,20,4,314)】】
                mini_batch = buffer.sample(min(len(buffer), conf.batch_size))
                # print(mini_batch['o'].shape)
                batch_change = {'o': [], 'u': [], 's': [], 'r': [], 'o_': [], 's_': [], 'avail_u': [],
                                'avail_u_': [], 'u_onehot': [], 'padded': [], 'terminated': []}
                for batch_data in mini_batch:
                    for key in batch_change.keys():
                        batch_change[key].append(batch_data[key][0])  # 把数据取出来按类型放入新的字典中
                        # batch_change[key] = np.array(batch_change[key])
                for itme in batch_change.keys():
                    batch_change[itme] = np.array(batch_change[itme])  # 把列表数据转换成数组
                agents.train(batch_change, train_steps)
                train_steps += 1
                print("当前已训练", train_steps, "次")

        # if epoch % conf.evaluate_per_epoch == 0:
        #     win_rate, episode_reward = evaluate(rollout_worker)
        #     win_rates.append(win_rate)
        #     episode_rewards.append(episode_reward)
        #     print("train epoch: {}, win rate: {}%, episode reward: {}".format(epoch, win_rate, episode_reward))
        # show_curves(win_rates, episode_rewards)

    # show_curves(win_rates, episode_rewards)


# def evaluate(rollout_worker):
#     # print("="*15, " evaluating ", "="*15)
#     win_num = 0
#     episode_rewards = 0
#     for epoch in range(conf.evaluate_epoch):
#         _, episode_reward, win_tag = rollout_worker.generate_episode(epoch, evaluate=True)
#         episode_rewards += episode_reward
#         if win_tag:
#             win_num += 1
#     return win_num / conf.evaluate_epoch, episode_rewards / conf.evaluate_epoch


# def show_curves(win_rates, episode_rewards):
#     print("=" * 15, " generate curves ", "=" * 15)
#     plt.figure()
#     plt.axis([0, conf.n_epochs, 0, 100])
#     plt.cla()
#     plt.subplot(2, 1, 1)
#     plt.plot(range(len(win_rates)), win_rates)
#     plt.xlabel('epoch*{}'.format(conf.evaluate_per_epoch))
#     plt.ylabel("win rate")
#
#     plt.subplot(2, 1, 2)
#     plt.plot(range(len(episode_rewards)), episode_rewards)
#     plt.xlabel('epoch*{}'.format(conf.evaluate_per_epoch))
#     plt.ylabel("episode reward")
#
#     plt.savefig(conf.result_dir + conf.map_name + '/result_plt.png', format='png')
#     np.save(conf.result_dir + conf.map_name + '/win_rates', win_rates)
#     np.save(conf.result_dir + conf.map_name + '/episode_rewards', episode_rewards)


if __name__ == "__main__":
    if conf.train:
        train()
