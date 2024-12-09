"""Hierarchical multi-agent RL with skill discovery.

Entry point for training HSD.
'Role' here is synonymous with 'skill' in the paper.
"""
# 导入基础库
import os
import time
import numpy as np
import torch
# 导入环境
import env.cpp_env.scenarios as scenarios
from env.cpp_env.environment import MultiAgentEnv
# 导入外部工作函数
from utils import create_dir
# from evaluate import evaluate
# 导入网络
from alg.qmix.agent import Agents as qmix_agents
from maddpg.agent import agents as maddpg_agents
from qmix.utils import ReplayBuffer as high_buffer
from maddpg.utils import ReplayBuffer as low_buffer
# 导入配置
from config import Config
from alg.arguments import parse_args
# 忽略tensorflow的警告
import warnings
import logging

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('numpy').setLevel(logging.ERROR)

conf = Config()


def make_env(arglist):
    """
    create the environment from script
    """
    scenario_name = arglist.scenario_name
    scenario = scenarios.load(scenario_name + ".py").Scenario(r_cover=0.1, r_comm=0.4, comm_r_scale=0.9,
                                                              comm_force_scale=5.0, env_size=10, arglist=arglist)
    world = scenario.make_world()
    if arglist.benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            done_callback=scenario.done)
    return env


def train_function(arg):
    """
        init the env, agent and train the agents
    """

    """step1: create the environment """
    env = make_env(arg)
    obs_shape_n = [env.observation_space[i].shape[0] + 1 for i in range(env.n)]
    action_shape_n = [env.action_space[i].n for i in range(env.n)]
    print('\n=============================')
    print('=1 Env {} is right ...'.format(arg.scenario_name))
    print('=============================')

    """step2: create agents"""
    # 定义各定义上下层网络网络
    high_agents = qmix_agents(conf)  # 上层策略
    low_agents = maddpg_agents(env, obs_shape_n, action_shape_n, arg)  # 下层策略

    # 定义上下层buffer
    buf_high = high_buffer(conf.buffer_size)
    buf_low = low_buffer(arg.buffer_size)

    # 初始化训练参数
    epsilon = conf.epsilon_start
    epsilon_end = conf.epsilon_end
    epsilon_step = conf.epsilon_step

    obs_size = []
    action_size = []
    all_step_h = 0
    train_steps_h = 0
    train_steps = 0

    # 保存模型评估时的数据
    test_reward = []
    test_coverage = []
    test_outRange = []
    test_collision = []
    test_done_steps = []

    coverage_rate = []
    start_time = time.time()

    # ------------------------------------ #
    # 开始训练场数的循环
    # ------------------------------------ #
    for idx_episode in range(arg.max_episode):

        # 初始化环境
        obs_n = env.reset()  # 得到所有智能体的观测
        obs_n_h = obs_n  # 带有后缀“_h”的变量用于训练高级策略
        state_h = np.concatenate(obs_n)

        # 初始化一场游戏的参数
        done = False
        done_n = [False for _ in range(conf.n_agents)]
        idx_step = 0  # 初始化当前步数
        reward_h = 0  # 上层策略累计折扣奖励
        reward_period = 0  # 所有场的训练的奖励累加
        episode_reward = 0  # 一场游戏的累计奖励
        rew_n = np.zeros(conf.n_agents)  # 初始化各个智能体的奖励

        # 上层qmix的相关初始化
        last_actions_h = np.zeros((conf.n_agents, conf.n_actions))  # [[0 0], [0, 0], ...] 初始化上一步的动作
        high_agents.policy.init_hidden(1)  # 初始化隐藏层状态，（1）表示是第一场
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []  # 记录轨迹
        idx_step_h = 0
        actions_h, actions_h_onehot, avail_actions_h = None, None, conf.avail_action
        # 开始训练步数循环
        while not done:
            # 初始化内在奖励
            reward_intrinsic = np.zeros(conf.n_agents)  # ndarray: [0.0  0.0  0.0  0.0]

            # --------------------------------------------------------------------------------------------------
            # 如果到达选择技能的步数.这里可以理解为上层策略的一个"step"
            # --------------------------------------------------------------------------------------------------
            if idx_step % conf.high_net_step == 0:
                actions_h, avail_actions_h, actions_h_onehot = [], [], []  # 初始化动作编码
                # 如果还在收集经验，则随机选择上层动作
                if idx_episode < conf.pretrain_episodes:
                    actions_h = np.random.randint(0, conf.n_actions, conf.n_agents)  # 【1, 0, 1, 1】
                    agent_id = 0
                    for action_h_single in actions_h:
                        action_h_onehot = np.zeros(conf.n_actions)  # [0 0]
                        action_h_onehot[action_h_single] = 1  # 表示选择动作1 【[0 1]】
                        actions_h_onehot.append(action_h_onehot)
                        avail_actions_h.append(conf.avail_action)
                        last_actions_h[agent_id] = action_h_onehot  # 把该智能体上次的动作记录下来
                        agent_id += 1

                else:
                    # 根据当前观测和贪婪系数，上层策略选择当前的潜在变量（上层动作）
                    # print(np.array(obs_n_h).shape, idx_step,high_agents.policy.eval_hidden.shape)
                    for agent_id in range(conf.n_agents):
                        # 选择动作
                        action_h_single = high_agents.choose_action(obs_n_h[agent_id], last_actions_h[agent_id],
                                                                    agent_id, conf.avail_action, epsilon)  # 0或者1
                        action_h_onehot = np.zeros(conf.n_actions)  # [0 0]
                        action_h_onehot[action_h_single] = 1  # 表示选择动作1 【[0 1]】
                        actions_h.append(action_h_single)  # 记录动作【[1, 0, ....]】
                        actions_h_onehot.append(action_h_onehot)  # 记录动作1hot【[[0 1], [1 0]], ....】
                        avail_actions_h.append(conf.avail_action)
                        last_actions_h[agent_id] = action_h_onehot  # 把该智能体上次的动作记录下来

                if idx_step != 0:
                    # 虽然对于下层来说进行了很多步，但是对于上层来说只进行了一步，所以折扣奖励是这样算的
                    r_step_h = reward_h * (conf.gamma ** conf.high_net_step)  # 计算这一步的上层折扣奖励

                    # 存储上层数据
                    obs_n_h_t = obs_n
                    state_t = np.concatenate(obs_n_h_t)
                    # reward_h = sum(rew_n)
                    done_h = False
                    if True in done_n:
                        done_h = True

                    o.append(obs_n_h_t)  # 观测
                    s.append(state_t)  #
                    # 每个时间步各个智能体的动作【[[[1], [0], [1], [1]], [[1], [0], [1], [1]], .......]】
                    u.append(np.reshape(actions_h, [conf.n_agents, 1]))
                    u_onehot.append(actions_h_onehot)  # 变成热编码 【[[[0, 1], [1 ,0], [0, 1], [0, 1]], [[0, 1], [1 ,0],
                    # [0, 1], [0, 1]], .......]】
                    avail_u.append(avail_actions_h)
                    r.append([r_step_h])  # 计算四个智能体的总奖励,以列表格式加入
                    terminate.append([done_h])  # 四个智能体总体的done
                    padded.append([0.])
                    # episode_reward += r_step_h  # 累加总奖励
                    if done_h:
                        break

                idx_step_h += 1
                all_step_h += 1  # 上层总步数加1

                # 用当前的状态更新上层状态，给下次使用
                obs_n_h = obs_n
                # 将一段轨迹的累积奖励置0
                reward_h = 0

            # --------------------------------------------------------------------------------------------------
            # 根据上层动作，来选择下层动作
            # --------------------------------------------------------------------------------------------------
            # 同样的，如果还在收集经验，则下层动作随机选择
            obs_n_l = []
            for i in range(len(actions_h)):
                obs_n_l.append(np.append(obs_n[i], actions_h[i]))

            if idx_episode < conf.pretrain_episodes:
                action_n_l = env.random_actions()
            else:
                # 基于观测和上层动作以及贪婪系数，下层策略选择动作
                action_n_l = [agent(torch.from_numpy(obs).to(arg.device, torch.float)).detach().cpu().numpy()
                              for agent, obs in zip(low_agents.actors_cur, obs_n_l)]
            # 基于环境得到下一时刻的观测，以及全局奖励，终止标志等。
            action_input = []
            for i in range(len(action_n_l)):
                action_input.append(np.append(action_n_l[i], actions_h[i]))

            obs_n_t, rew_n, done_n = env.step(action_input)
            if idx_episode % arg.render_fra == 0 and idx_step % arg.render_step_fra:
                env.render()

            obs_n_t_l = []
            for i in range(len(actions_h)):
                obs_n_l.append(np.append(obs_n[i], actions_h[i]))

            # 将下层策略的数据打包存储到下层经验池中
            buf_low.add(obs_n_l, np.concatenate(action_n_l), rew_n, obs_n_t_l, done_n)
            idx_step += 1  # 当前总步数加1

            # 如果经验采集完毕，且到达下层策略的训练周期
            if (idx_episode >= conf.pretrain_episodes) and (idx_step % arg.steps_per_train == 0):
                # 训练下层策略
                (train_steps, low_agents.actors_cur, low_agents.actors_tar, low_agents.critics_cur,
                 low_agents.critics_tar) = low_agents.agents_train(arg, idx_step, train_steps, buf_low, obs_size,
                                                                   action_size)
            # 更新状态和观测，以及当前游戏的奖励reward_episode，还有后面要给上层用来训练的上层奖励reward_h
            obs_n = obs_n_t
            done = all(done_n)
            episode_reward += sum(rew_n)  # 累积这一整局的奖励
            reward_h += sum(rew_n)  # 为上层累积一个step_h的奖励
            cov = env.world.coverage_rate
            coverage_rate.append(cov)
            # 达到最大步数，终止游戏
            if idx_step >= arg.max_step:
                done = True

            if done:
                # 由于该集已经完成，所以同时也终止上层动作的选择，即使还没有到上层动作的周期<steps_per_assign>
                r_step_h = reward_h * arg.gamma ** (idx_step % arg.high_net_step)  # 当前的折扣奖励
                # 存储上层数据
                obs_n_h_t = obs_n
                state_t = np.concatenate(obs_n_h_t)
                # reward_h = sum(rew_n)

                o.append(obs_n_h_t)  # 观测
                s.append(state_t)
                # 每个时间步各个智能体的动作【[[[1], [0], [1], [1]], [[1], [0], [1], [1]], .......]】
                u.append(np.reshape(actions_h, [conf.n_agents, 1]))
                u_onehot.append(actions_h_onehot)  # 变成热编码 【[[[0, 1], [1 ,0], [0, 1], [0, 1]], [[0, 1], [1 ,0],
                # [0, 1], [0, 1]], .......]】
                avail_u.append(avail_actions_h)
                r.append([r_step_h])  # 计算四个智能体的总奖励,以列表格式加入
                terminate.append([done])  # 四个智能体总体的done
                padded.append([0.])
                idx_step_h += 1

        # 如果经验采集完毕，并且当前步数达到训练周期
        if (idx_episode >= conf.pretrain_episodes) and (idx_episode % conf.train_frequency == 0):
            for train_step in range(conf.train_steps):
                mini_batch = buf_high.sample(min(len(buf_high), conf.batch_size))
                batch_change = {'o': [], 'u': [], 's': [], 'r': [], 'o_': [], 's_': [], 'avail_u': [],
                                'avail_u_': [], 'u_onehot': [], 'padded': [], 'terminated': []}
                for batch_data in mini_batch:
                    for key in batch_change.keys():
                        batch_change[key].append(batch_data[key][0])  # 把数据取出来按类型放入新的字典中
                        # batch_change[key] = np.array(batch_change[key])
                for itme in batch_change.keys():
                    batch_change[itme] = np.array(batch_change[itme])  # 把列表数据转换成数组
                high_agents.train(batch_change, train_steps_h)
                train_steps_h += 1
                print("上层网络当前已训练", train_steps_h, "次")

        # 最后一个动作
        o.append(obs_n)
        s.append(np.concatenate(obs_n))
        o_ = o[1:]
        s_ = s[1:]
        o = o[:-1]
        s = s[:-1]

        # target q 在last obs需要avail_action
        avail_actions = []
        for agent_id in range(conf.n_agents):
            avail_actions.append(conf.avail_action)
        avail_u.append(avail_actions)
        avail_u_ = avail_u[1:]
        avail_u = avail_u[:-1]

        # 当step<self.episode_limit时，输入数据加padding
        for i in range(idx_step_h, int(arg.max_step/conf.high_net_step)+1):
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

        buf_high.add(episode)
        print("当前已运行", idx_episode, "场")
        if idx_episode >= conf.pretrain_episodes and idx_episode % conf.train_frequency == 0:  # 每10场训练一次
            for train_step in range(conf.train_steps):  # 一次训练多少轮
                # 一个mini_batch有batch_size个数据每个数据中有11类数据，【其中obs:【一场游戏，20个时间步，四个智能体，314位观测(1,20,4,314)】】
                mini_batch = buf_high.sample(min(len(buf_high), conf.batch_size))
                # print(mini_batch['o'].shape)
                batch_change = {'o': [], 'u': [], 's': [], 'r': [], 'o_': [], 's_': [], 'avail_u': [],
                                'avail_u_': [], 'u_onehot': [], 'padded': [], 'terminated': []}
                for batch_data in mini_batch:
                    for key in batch_change.keys():
                        batch_change[key].append(batch_data[key][0])  # 把数据取出来按类型放入新的字典中
                        # batch_change[key] = np.array(batch_change[key])
                for itme in batch_change.keys():
                    batch_change[itme] = np.array(batch_change[itme])  # 把列表数据转换成数组
                high_agents.train(batch_change, train_steps)
                train_steps += 1
                print("当前已训练", train_steps, "次")

        # 累加这一集的奖励
        reward_period += episode_reward

        if idx_episode > 0 and idx_episode % arg.print_fre == 0:  # 一定间隔尽心打印一次（评估中每训练5次【其实不用】）
            # print('{:>10s}{:>10s}{:>12s}{:>5s}{:>8s}{:>10s}{:>8s}{:>8s}{:>15s}{:>15s}{:>10s}{:>12s}{:>12s}'.format(
            #     *(header.strip().split(','))))  # 打印result信息的头header,即每五次打印一次头header
            end_time = time.time()
            output = format("Eps: %.1fk, Rew: %.2f, Cov:%.2f "
                            "time: %.2fs"
                            % (idx_episode / 1000,
                               float(reward_period / idx_episode),
                               float(np.mean(coverage_rate[-arg.print_fre:])),
                               end_time - start_time
                               ))
            print(output)
        if idx_episode >= conf.pretrain_episodes and epsilon > epsilon_end:
            epsilon -= epsilon_step

        if idx_episode > 0 and idx_episode % arg.fre_save_model == 0:  # 如果到了保存间隔，不论胜率是否达标都保存
            time_now = time.strftime('%m%d_%H%M%S')
            print('=time:{} episode:{} step:{}        save'.format(time_now, idx_episode + 1, idx_step))
            model_file_dir = os.path.join(save_policy_path, '{}_{}_{}'.format(
                arg.scenario_name, time_now, (idx_episode + 1)))
            if not os.path.exists(model_file_dir):  # make the path
                os.mkdir(model_file_dir)
            for agent_idx, (a_c, a_t, c_c, c_t) in \
                    enumerate(zip(low_agents.actors_cur, low_agents.actors_tar, low_agents.critics_cur,
                                  low_agents.critics_tar)):
                torch.save(a_c, os.path.join(model_file_dir, 'a_c_{}.pt'.format(agent_idx)))
                torch.save(a_t, os.path.join(model_file_dir, 'a_t_{}.pt'.format(agent_idx)))
                torch.save(c_c, os.path.join(model_file_dir, 'c_c_{}.pt'.format(agent_idx)))
                torch.save(c_t, os.path.join(model_file_dir, 'c_t_{}.pt'.format(agent_idx)))

    # 结束完训练，进行保存
    time_now = time.strftime('%m%d_%H%M%S')
    model_file_dir = os.path.join(save_policy_path, '{}_{}_{}'.format(
        arg.scenario_name, time_now, (arg.max_episode + 1)))
    if not os.path.exists(model_file_dir):  # make the path
        os.mkdir(model_file_dir)


if __name__ == '__main__':
    arg_list = parse_args()
    save_policy_path, save_plots_path = create_dir(arg_list.scenario_name)
    train_function(arg_list)
