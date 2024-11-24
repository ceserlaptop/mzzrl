"""Hierarchical multi-agent RL with skill discovery.

Entry point for training HSD.
'Role' here is synonymous with 'skill' in the paper.
"""

import json
import os
import random
import sys
import time
# sys.path.append('../env/')
import numpy as np
import tensorflow as tf
import alg_hsd
from env import env_wrapper
import evaluate
import replay_buffer
# 忽略tensorflow的警告
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('numpy').setLevel(logging.ERROR)


def train_function(config):
    # 参数的传递
    config_env = config['env']
    config_main = config['main']
    config_alg = config['alg']
    config_h = config['h_params']
    # 随机种子的初始化
    seed = config_main['seed']
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    # 算法参数的初始化
    alg_name = config_main['alg_name']
    dir_name = config_main['dir_name']
    model_name = config_main['model_name']
    summarize = config_main['summarize']
    save_period = config_main['save_period']

    os.makedirs('../results/%s' % dir_name, exist_ok=True)
    with open('../results/%s/%s'
              % (dir_name, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    N_train = config_alg['N_train']  # 总的训练运行轮数
    N_eval = config_alg['N_eval']  # 评估多少轮
    period = config_alg['period']  # 评估期间的训练的轮数【不用，因为不总结】，同时还代表打印轮数
    buffer_size = config_alg['buffer_size']
    batch_size = config_alg['batch_size']
    pretrain_episodes = config_alg['pretrain_episodes']  # 训练开始前使用随机动作采集数据的轮数
    steps_per_train = config_alg['steps_per_train']  # 每隔多少步训练一次

    epsilon_start = config_alg['epsilon_start']  # Q学习的贪婪系数最大值
    epsilon_end = config_alg['epsilon_end']  # Q学习的贪婪系数最小值
    epsilon_div = config_alg['epsilon_div']  # Q学习的贪婪系数的下降轮数
    epsilon_step = (epsilon_start - epsilon_end) / float(epsilon_div)  # 每一轮贪婪系数下降值
    epsilon = epsilon_start

    # Final number of roles
    N_roles = config_h['N_roles']  # 技能的最大数量
    steps_per_assign = config_h['steps_per_assign']  # 搁多少步选一次技能

    # Number of roles increases according to a curriculum
    N_roles_current = config_h['N_roles_start']  # 初始的技能数量，要等于N_roles，因为目前还不支持增加技能数量
    assert (N_roles_current <= N_roles)  # 确保N_roles_current不超过N_roles
    curriculum_threshold = config_h['curriculum_threshold']  # 当解码器性能超过此阈值时，技能数量会增加（目前不支持）

    # Reward coefficient
    alpha = config_h['alpha_start']  # 衡量内在和外在奖励的系数的初始值
    alpha_end = config_h['alpha_end']
    alpha_step = config_h['alpha_step']
    alpha_threshold = config_h['alpha_threshold']

    # Number of single-agent trajectory segments used for each decoder training step 
    N_batch_hsd = config_h['N_batch_hsd']

    env = env_wrapper.Env(config_env, config_main)
    # 获取动作空间、状态空间、观测空间的维度
    l_state = env.state_dim
    l_action = env.action_dim
    l_obs = env.obs_dim
    N_home = config_env['num_home_players']  # 合作智能体的数量

    # 定义智能体的网络
    alg = alg_hsd.Alg(config_alg, config_h, N_home, l_state, l_obs, l_action, N_roles, config['nn_hsd'])

    # 配置tf会话和内存
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())

    sess.run(alg.list_initialize_target_ops)

    if summarize:
        writer = tf.summary.FileWriter('../results/%s' % dir_name, sess.graph)
    saver = tf.train.Saver(max_to_keep=config_main['max_to_keep'])

    # Replay buffer for high level policy
    buf_high = replay_buffer.Replay_Buffer(size=buffer_size)
    # Buffer for low level agent policy
    buf_low = replay_buffer.Replay_Buffer(size=buffer_size)

    # Dataset of [obs traj, z] for training decoder
    dataset = []

    # header = "Episode,Step,Step_train,N_z,alpha,Exp_prob,R_avg,R_eval,Steps_per_eps,Opp_win_rate,Win_rate,T_env,
    # T_alg\n"
    header = "当前集数,步数,训练步数,当前技巧数,alpha,编码器预测概率,平均奖励,评估的平均奖励,每集平均步长,敌方胜率,我方胜率,环境运行时间,算法运行时间\n"
    with open("../results/%s/log.csv" % dir_name, 'w') as f:
        f.write(header)

    # 初始化训练参数
    t_env = 0
    t_alg = 0

    reward_period = 0
    expected_prob = 0

    step = 0
    step_train = 0
    step_h = 0
    # 开始训练轮数的循环
    for idx_episode in range(1, N_train + 1):
        # 主队状态，客队状态，主队观测，客队观测，是否结束
        state_home, state_away, list_obs_home, list_obs_away, done = env.reset()

        # 带有后缀“_h”的变量用于训练高级策略
        state_home_h, state_away_h, list_obs_home_h, list_obs_away_h = state_home, state_away, list_obs_home, list_obs_away
        # 上层策略累计折扣奖励
        reward_h = 0
        # 上层政策采取的行动，仅限于当前课程中的角色数量（初始化是随机初始化的）
        roles_int = np.random.randint(0, N_roles_current, N_home)
        roles = None
        # 列表的列表，其中每个子列表是代理的观察轨迹
        # 每个轨迹最多包含长度为 <steps_per_assign> 的先前步骤的历史记录，
        # 用于计算内在奖励
        list_obs_traj = [[] for idx_agent in range(N_home)]

        reward_episode = 0
        summarized = 0
        summarized_h = 0
        step_episode = 0  # steps within an episode

        # 开始一局的游戏中的训练步数循环
        while not done:
            # 初始化内在奖励
            reward_intrinsic = np.zeros(N_home)
            # --------------------------------------------------------------------------------------------------
            # 如果到达选择技能的步数.这里可以理解为上层策略的一个"step"
            # --------------------------------------------------------------------------------------------------
            if step_episode % steps_per_assign == 0:
                if step_episode != 0:
                    # 此时的环境状态， e.g. <state_home>,
                    # 相当于上层政策的“下一个状态”（下一个技能）
                    # 虽然对于下层来说进行了很多步，但是对于上层来说，只进行了一步，所以折扣奖励是这样算的
                    # All of the intervening environment steps act as a single step for the high-level policy

                    r_discounted = reward_h * (config_alg['gamma'] ** steps_per_assign)  # 计算折扣奖励
                    # 将上层状态和观测（一个时间间隔前的）、动作、折扣奖励、下层状态和观测（对于上层来说是下一时刻的状态和观测）、是否结束加入到上层buffer中
                    buf_high.add(np.array([state_home_h, np.array(list_obs_home_h), roles_int, r_discounted, state_home,
                                           np.array(list_obs_home), done]))

                    # 将所有代理的观察轨迹存储到数据集中用于训练解码器
                    for idx in range(N_home):
                        # 存储潜在变量和对应的轨迹
                        dataset.append(np.array([np.array(list_obs_traj[idx][-steps_per_assign:]), roles[idx]]))

                    # 使用编码器计算所有代理的内在奖励， returns an np.array of N_agents scalar values
                    reward_intrinsic = alg.compute_reward(sess, np.array(list_obs_traj), roles)

                step_h += 1

                # 选择潜在变量
                # 如果还在收集经验，则随机选择上层动作
                if idx_episode < pretrain_episodes:
                    roles_int = np.random.randint(0, N_roles_current, N_home)
                else:
                    t_alg_start = time.time()
                    # 根据当前观测和贪婪系数，上层策略选择当前的潜在变量（上层动作）
                    roles_int = alg.assign_roles(list_obs_home, epsilon, sess, N_roles_current)
                    t_alg += time.time() - t_alg_start  # 计算算法选择动作所花费的时间
                roles = np.zeros([N_home, N_roles])  # 初始化为全0
                # 得到上层动作
                roles[np.arange(N_home), roles_int] = 1  # 根据选择的潜在变量，将相应位置置为1.给每个智能体分配到特定的技能
                # 如果经验采集完毕，并且当前步数达到训练周期
                if (idx_episode >= pretrain_episodes) and (step_h % steps_per_train == 0):
                    # 开始训练上层策略
                    batch = buf_high.sample_batch(batch_size)  # 获取一个batch的数据
                    t_alg_start = time.time()
                    if summarize and idx_episode % period == 0 and not summarized_h:
                        alg.train_policy_high(sess, batch, step_train, summarize=True, writer=writer)
                        summarized_h = True
                    else:
                        # 上层策略进行训练
                        alg.train_policy_high(sess, batch, step_train, summarize=False, writer=None)
                    step_train += 1
                    t_alg += time.time() - t_alg_start  # 计算算法所花费的时间

                # 用当前的状态更新上层状态，给下次使用
                state_home_h, state_away_h, list_obs_home_h, list_obs_away_h = state_home, state_away, list_obs_home, list_obs_away
                # 将一段轨迹的累积奖励置0
                reward_h = 0

            # --------------------------------------------------------------------------------------------------
            # 根据上层动作，来选择下层动作
            # --------------------------------------------------------------------------------------------------
            # 同样的，如果还在收集经验，则下层动作随机选择
            if idx_episode < pretrain_episodes:
                actions_int = env.random_actions()
            else:
                t_alg_start = time.time()
                # 基于观测和上层动作以及贪婪系数，下层策略选择动作
                actions_int = alg.run_actor(list_obs_home, roles, epsilon, sess)
                t_alg += time.time() - t_alg_start

            t_env_start = time.time()
            # 基于环境得到下一时刻的主队和客队的状态和观测，以及全局奖励和局部奖励（这个局部奖励后买你不用），终止标志等。
            state_home_next, state_away_next, list_obs_home_next, list_obs_away_next, reward, local_rewards, done, info \
                = env.step(actions_int)
            t_env += time.time() - t_env_start

            # 忽略环境中的 local_rewards重新写。使用内在和全局环境奖励计算局部奖励。
            local_rewards = np.array([reward] * N_home)
            local_rewards = alpha * local_rewards + (1 - alpha) * reward_intrinsic

            # 收集下层观测，记录成轨迹
            for idx_agent in range(N_home):
                list_obs_traj[idx_agent].append(list_obs_home[idx_agent])
                # 限制轨迹的最大长度，即每次选择上层动作的间隔  Limit to be max length <steps_per_assign>
                list_obs_traj[idx_agent] = list_obs_traj[idx_agent][-steps_per_assign:]

            step += 1
            step_episode += 1
            # 将下层策略的数据打包存储到下层经验池中
            l_temp = [np.array(list_obs_home), actions_int, local_rewards, np.array(list_obs_home_next), roles, done]
            a_temp = np.empty(len(l_temp), dtype=object)  # 创建一个与l_temp相同长度的数组
            a_temp[:] = l_temp  # 把l_temp的内容复制到a_temp
            buf_low.add(a_temp)  # 存储经验

            # 如果经验采集完毕，且到达下层策略的训练周期
            if (idx_episode >= pretrain_episodes) and (step % steps_per_train == 0):
                # 训练下层策略
                batch = buf_low.sample_batch(batch_size)
                t_alg_start = time.time()
                if summarize and idx_episode % period == 0 and not summarized:
                    alg.train_policy_low(sess, batch, step_train, summarize=True, writer=writer)
                    summarized = True
                else:
                    alg.train_policy_low(sess, batch, step_train, summarize=False, writer=None)
                step_train += 1
                t_alg += time.time() - t_alg_start

            # 更新状态和观测，以及当前游戏的奖励reward_episode，还有后面要给上层用来训练的上层奖励reward_h
            state_home = state_home_next
            list_obs_home = list_obs_home_next
            reward_episode += reward
            reward_h += reward

            if done:
                # 由于该集已经完成，所以同时也终止上层动作的选择，即使还没有到上层动作的周期<steps_per_assign>
                r_discounted = reward_h * config_alg['gamma'] ** (step_episode % steps_per_assign)  # 当前的折扣奖励
                buf_high.add(np.array([state_home_h, np.array(list_obs_home_h), roles_int, r_discounted, state_home,
                                       np.array(list_obs_home), done]))  # 将这个数据也加入上层经验池中
                # 将轨迹附加到数据集中，以便解码器看到获得终止奖励的状态
                # Append trajectories into dataset, so that decoder sees states that get termination reward
                if step_episode >= steps_per_assign:
                    for idx in range(N_home):
                        dataset.append(np.array([np.array(list_obs_traj[idx][-steps_per_assign:]), roles[idx]]))

        # 如果数据集足够大，则训练解码器
        if len(dataset) >= N_batch_hsd:
            t_alg_start = time.time()
            if summarize:
                expected_prob = alg.train_decoder(sess, dataset[: N_batch_hsd], step_train, summarize=True,
                                                  writer=writer)
            else:
                expected_prob = alg.train_decoder(sess, dataset[: N_batch_hsd], step_train, summarize=False,
                                                  writer=None)
            step_train += 1
            t_alg += time.time() - t_alg_start
            # 决定是否增加技巧（上层动作）的数量 Decide whether to increase the number of subgoals
            if expected_prob >= curriculum_threshold:  # 如果这个技巧太熟悉了，可以再细化，多增加几个技巧
                N_roles_current = min(int(1.5 * N_roles_current + 1), N_roles)  # 确保增加单不会大于N_roles
            # 清空数据集
            dataset = []
        # 如果采集完经验且贪婪系数还没到最小，则逐步减小贪婪系数的值
        if idx_episode >= pretrain_episodes and epsilon > epsilon_end:
            epsilon -= epsilon_step

        # 累加这一集的奖励
        reward_period += reward_episode

        if idx_episode == 1 or idx_episode % (5 * period) == 0:  # 一定间隔尽心打印一次（评估中每训练5次【其实不用】）
            print('{:>10s}{:>10s}{:>12s}{:>5s}{:>8s}{:>10s}{:>8s}{:>8s}{:>15s}{:>15s}{:>10s}{:>12s}{:>12s}'.format(
                *(header.strip().split(','))))  # 打印result信息的头header,即每五次打印一次头header

        if idx_episode % period == 0:
            # 进行评估，得到平均奖励，每集平均步长，胜率，对手胜率
            r_avg_eval, steps_per_episode, win_rate, win_rate_opponent = evaluate.test_hierarchy(alg_name, N_eval, env,
                                                                                                 sess,
                                                                                                 alg,steps_per_assign)
            # 如果胜率大于保存的阈值，则保存策略
            if win_rate >= config_main['save_threshold']:
                saver.save(sess, '../results/%s/%s-%d' % (dir_name, "model_good.ckpt", idx_episode))

            # 调整环境奖励与内在奖励的alpha系数
            if win_rate >= alpha_threshold:
                alpha = max(alpha_end, alpha - alpha_step)

            s = '%d,%d,%d,%d,%.2f,%.3e,%.2f,%.2f,%d,%.2f,%.2f,%.5e,%.5e\n' % (
                idx_episode, step, step_train, N_roles_current, alpha, expected_prob, reward_period / float(period),
                r_avg_eval, steps_per_episode, win_rate_opponent, win_rate, t_env, t_alg)
            with open('../results/%s/log.csv' % dir_name, 'a') as f:
                f.write(s)  # 记录到log.csv文件中

            # 打印到终端
            print('{:10d}{:10d}{:12d}{:5d}{:8.2f}{:10.3e}{:8.2f}{:8.2f}{:15d}{:15.2f}{:10.2f}{:12.5e}{:12.5e}\n'.format(
                idx_episode, step, step_train, N_roles_current, alpha, expected_prob, reward_period / float(period),
                r_avg_eval, int(steps_per_episode), win_rate_opponent, win_rate, t_env, t_alg))
            reward_period = 0  # 这次评估完后，将奖励置0，重新累加来评估

        if idx_episode % save_period == 0:  # 如果到了保存间隔，不论胜率是否达标都保存
            saver.save(sess, '../results/%s/%s-%d' % (dir_name, "model.ckpt", idx_episode))
    # 结束完训练，进行保存
    saver.save(sess, '../results/%s/%s' % (dir_name, model_name))
    # 最后记录到log.csv中
    with open('../results/%s/time.txt' % dir_name, 'a') as f:
        f.write('t_env_total,t_env_per_step,t_alg_total,t_alg_per_step\n')
        f.write('%.5e,%.5e,%.5e,%.5e' % (t_env, t_env / step, t_alg, t_alg / step))


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)
    train_function(config)
