import math
import os
import pickle
import time
import torch
# torch.backends.cudnn.enabled = False
import numpy as np
import torch.nn as nn
import torch.optim as optim
from arguments import parse_args
from replay_buffer import ReplayBuffer
import sys
import multiagent_env.multiagent.scenarios as scenarios
from model import openai_actor, openai_critic
from multiagent_env.multiagent.environment import MultiAgentEnv
import torch.nn.functional as F
sys.path.append(os.path.abspath("../multiagent_env/"))

scenario_name = "coverage_0"
save_policy_path = None
save_plots_path = None


def make_env(scenario_name, arglist):
    """
    create the environment from script
    """
    scenario = scenarios.load(scenario_name + ".py").Scenario(r_cover=0.1, r_comm=0.4, comm_r_scale=0.9,
                                                              comm_force_scale=5.0, env_size=10, arglist=arglist)
    world = scenario.make_world()
    if arglist.benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            done_callback=scenario.done)
        # env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def get_trainers(env, obs_shape_n, action_shape_n, arglist):
    """
    init the trainers or load the old model
    """
    actors_cur = [None for _ in range(env.n)]
    critics_cur = [None for _ in range(env.n)]
    actors_tar = [None for _ in range(env.n)]
    critics_tar = [None for _ in range(env.n)]
    optimizers_c = [None for _ in range(env.n)]
    optimizers_a = [None for _ in range(env.n)]

    if arglist.restore:  # restore the model
        for i in range(env.n):
            actors_cur[i] = (torch.load(arglist.old_model_name + 'a_c_{}'.format(i))).to(arglist.device)
            critics_cur[i] = (torch.load(arglist.old_model_name + 'c_c_{}'.format(i))).to(arglist.device)
            actors_tar[i] = (torch.load(arglist.old_model_name + 'a_t_{}'.format(i))).to(arglist.device)
            critics_tar[i] = (torch.load(arglist.old_model_name + 'c_t_{}'.format(i))).to(arglist.device)
            optimizers_a[i] = optim.Adam(actors_cur[i].parameters(), arglist.lr_a)
            optimizers_c[i] = optim.Adam(critics_cur[i].parameters(), arglist.lr_c)
    else:
        for i in range(env.n):
            actors_cur[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
            critics_cur[i] = openai_critic(sum(obs_shape_n), sum(action_shape_n), arglist).to(arglist.device)
            actors_tar[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
            critics_tar[i] = openai_critic(sum(obs_shape_n), sum(action_shape_n), arglist).to(arglist.device)
            optimizers_a[i] = optim.Adam(actors_cur[i].parameters(), arglist.lr_a)
            optimizers_c[i] = optim.Adam(critics_cur[i].parameters(), arglist.lr_c)
    actors_tar = update_train_tar(actors_cur, actors_tar, 1.0)  # update the target par using the cur
    critics_tar = update_train_tar(critics_cur, critics_tar, 1.0)  # update the target par using the cur
    return actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c


def update_train_tar(agents_cur, agents_tar, tau):
    """
    update the trainers_tar par using the trainers_cur
    This way is not the same as copy_, but the result is the same
    out:
    |agents_tar: the agents with new par updated towards agents_current
    """
    for agent_c, agent_t in zip(agents_cur, agents_tar):
        key_list = list(agent_c.state_dict().keys())
        state_dict_t = agent_t.state_dict()
        state_dict_c = agent_c.state_dict()
        for key in key_list:
            state_dict_t[key] = state_dict_c[key] * tau + \
                                (1 - tau) * state_dict_t[key]
        agent_t.load_state_dict(state_dict_t)
    return agents_tar


def agents_train(arglist, game_step, update_cnt, memory, obs_size, action_size, actors_cur, actors_tar, critics_cur,
                 critics_tar, optimizers_a, optimizers_c):
    """
    use this func to make the "main" func clean
    par:
    |input: the data for training
    |output: the data for next update
    """
    # update all trainers, if not in display or benchmark mode
    if game_step > arglist.learning_start_step and \
            (game_step - arglist.learning_start_step) % arglist.learning_fre == 0:
        if update_cnt == 0:
            print('Start training ...')
        # update the target par using the cur
        update_cnt += 1

        # update every agent in different memory batch
        for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) in \
                enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)):
            # sample the experience
            _obs_n, _action_n, _rew_n, _obs_n_t, _done_n = memory.sample(
                arglist.batch_size, agent_idx)  # Note_The func is not the same as others

            # --use the data to update the CRITIC
            # set the data to gpu
            rew = torch.tensor(_rew_n, dtype=torch.float, device=arglist.device)
            done_n = torch.tensor(~_done_n, dtype=torch.float, device=arglist.device)
            action_n = torch.from_numpy(_action_n).to(arglist.device, torch.float)
            obs_n = torch.from_numpy(_obs_n).to(arglist.device, torch.float)
            obs_n_t = torch.from_numpy(_obs_n_t).to(arglist.device, torch.float)
            # cal the loss
            action_tar = torch.cat([a_t(obs_n_t[:, obs_size[idx][0]:obs_size[idx][1]]).detach()
                                    for idx, a_t in enumerate(actors_tar)], dim=1)  # get the action in next state
            q = critic_c(obs_n, action_n).reshape(-1)  # q value in current state
            q_t = critic_t(obs_n_t, action_tar).reshape(-1)  # q value in next state
            tar_value = q_t * arglist.gamma * done_n + rew  # q_*gamma*done + reward
            loss_c = torch.nn.MSELoss()(q, tar_value)  # bellman equation
            # update the parameters
            opt_c.zero_grad()
            loss_c.backward()
            nn.utils.clip_grad_norm_(critic_c.parameters(), arglist.max_grad_norm)
            opt_c.step()

            # --use the data to update the ACTOR
            # There is no need to cal other agent's action
            model_out, policy_c_new = actor_c(
                obs_n[:, obs_size[agent_idx][0]:obs_size[agent_idx][1]], model_original_out=True)
            # update the action of this agent
            action_n[:, action_size[agent_idx][0]:action_size[agent_idx][1]] = policy_c_new
            loss_pse = torch.mean(torch.pow(model_out, 2))
            loss_a = torch.mul(-1, torch.mean(critic_c(obs_n, action_n)))

            opt_a.zero_grad()
            (1e-3 * loss_pse + loss_a).backward()
            nn.utils.clip_grad_norm_(actor_c.parameters(), arglist.max_grad_norm)
            opt_a.step()

        # update the target network
        actors_tar = update_train_tar(actors_cur, actors_tar, arglist.tau)
        critics_tar = update_train_tar(critics_cur, critics_tar, arglist.tau)

    return update_cnt, actors_cur, actors_tar, critics_cur, critics_tar


def mytest_model(arglist, actors_cur, num_test_episode):
    # 创建环境
    env = make_env(arglist.scenario_name, arglist)
    obs_n = env.reset()

    episode_coverage = []
    episode_outRange = [0]
    episode_collision = [0]
    episode_rewards = [0]
    done_steps = []

    for test_episode in range(num_test_episode):
        episode_step = 0
        while True:
            # get action
            action_n = []
            for actor, obs in zip(actors_cur, obs_n):
                model_out, _ = actor(torch.from_numpy(obs).to(arglist.device, torch.float), model_original_out=True)
                action_n.append(F.softmax(model_out, dim=-1).detach().cpu().numpy())

            # interact with env
            new_obs_n, rew_n, done_n = env.step(action_n)
            obs_n = new_obs_n

            if env.world.collision:
                episode_collision[-1] += 1
            if env.world.outRange:
                episode_outRange[-1] += 1

            episode_step += 1
            terminal = (episode_step >= arglist.max_step_len) or all(done_n)
            if terminal:
                episode_outRange[-1] /= episode_step
                episode_outRange.append(0)
                episode_collision[-1] /= episode_step
                episode_collision.append(0)
                cov = env.world.coverage_rate
                episode_coverage.append(cov)
                episode_rewards.append(np.sum(rew_n))
                if env.world.coverage_rate == 1.0:
                    done_steps.append(episode_step)
                else:
                    done_steps.append(arglist.max_step_len)
                obs_n = env.reset()
                break

    return episode_coverage, episode_rewards[:-1], episode_collision[:-1], episode_outRange[-1], done_steps


def train(arglist):
    """
    init the env, agent and train the agents
    """
    """step1: create the environment """
    env = make_env(arglist.scenario_name, arglist)

    print('=============================')
    print('=1 Env {} is right ...'.format(arglist.scenario_name))
    print('=============================')

    """step2: create agents"""
    # n denotes agent numbers
    obs_shape_n = [env.observation_space[i].shape[0] for i in range(env.n)]
    action_shape_n = [env.action_space[i].n for i in range(env.n)]  # no need for stop bit
    actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c = \
        get_trainers(env, obs_shape_n, action_shape_n, arglist)
    memory = ReplayBuffer(arglist.memory_size)

    print('=2 The {} agents are inited ...'.format(env.n))
    print('=============================')

    """step3: init the pars """
    obs_size = []
    action_size = []
    agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward to train model

    # 保存模型训练时的数据
    game_step = 0  # game steps until now for all episode
    update_cnt = 0  # numbers of updating the models
    episode_rewards = []  # sum of rewards for all agents in per
    episode_collision = [0]
    episode_outRange = [0]
    coverage_rate = []
    done_steps = []
    modify_done_steps = []

    # 保存模型评估时的数据
    test_reward = []
    test_coverage = []
    test_outRange = []
    test_collision = []
    test_done_steps = []

    start_time = time.time()
    obs_n = env.reset()
    evaluation_fre = 100
    print_fre = 100
    episode_gone = 0

    head_o, head_a, end_o, end_a = 0, 0, 0, 0
    for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
        end_o = end_o + obs_shape
        end_a = end_a + action_shape
        range_o = (head_o, end_o)
        range_a = (head_a, end_a)
        obs_size.append(range_o)
        action_size.append(range_a)
        head_o = end_o
        head_a = end_a

    print('=3 starting iterations ...')
    print('=============================')
    while True:
        # cal the reward print the debug data  计算奖励并打印调试数据
        if game_step > 1 and episode_gone % print_fre == 0:
            end_time = time.time()
            output = format("Steps: %d, Eps: %.1fk, Rew: %.2f, States: %d, Cov:%.2f, Col:%.3f, Out: %.3f, "
                            "time: %.2fs"
                            % (game_step,
                               episode_gone / 1000,
                               float(np.mean(episode_rewards[-print_fre:])),
                               float(np.mean(done_steps[-print_fre:])),
                               float(np.mean(coverage_rate[-print_fre:])),
                               float(np.mean(episode_collision[-print_fre:-1])),
                               float(np.mean(episode_outRange[-print_fre:-1])),
                               end_time - start_time
                               ))
            print(output)
            with open(save_plots_path + '/log.txt', 'a') as f:
                f.write(output + '\n')

        game_step_cur = 0
        # 开始一局游戏的循环
        for episode_cnt in range(arglist.max_step):
            # get action  选择动作
            action_n = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy()
                        for agent, obs in zip(actors_cur, obs_n)]
            # interact with env
            obs_n_t, rew_n, done_n = env.step(action_n)

            # save the experience
            memory.add(obs_n, np.concatenate(action_n), rew_n, obs_n_t, done_n)
            for i, rew in enumerate(rew_n):
                agent_rewards[i][-1] += rew

            # train our agents
            update_cnt, actors_cur, actors_tar, critics_cur, critics_tar = agents_train(
                arglist, game_step, update_cnt, memory, obs_size, action_size,
                actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)

            if episode_gone % 1 == 0:
                # 每100轮且100轮的每10步才render一次
                env.render()
            time.sleep(0.1)

            # update the obs_n
            game_step += 1
            game_step_cur += 1
            obs_n = obs_n_t
            # if env.world.collision:
            #     episode_collision[-1] += 1
            # if env.world.outRange:
            #     episode_outRange[-1] += 1

            done = all(done_n)
            terminal = (episode_cnt >= arglist.max_step - 1)
            cov = env.world.coverage_rate
            if done or terminal:
                episode_outRange[-1] /= game_step_cur
                episode_outRange.append(0)
                episode_collision[-1] /= game_step_cur
                episode_collision.append(0)
                episode_rewards.append(np.sum(rew_n))
                coverage_rate.append(cov)
                if cov == 1.0:
                    modify_done_steps.append(game_step_cur)
                else:
                    modify_done_steps.append(arglist.max_step)
                done_steps.append(game_step_cur)
                obs_n = env.reset()
                for a_r in agent_rewards:
                    a_r.append(0)
                break

        if episode_gone > 0 and episode_gone % evaluation_fre == 0:
            test_cov, test_rew, test_col, test_out, test_done = mytest_model(arglist, actors_cur, evaluation_fre)
            test_coverage.append((np.mean(test_cov)))
            test_reward.append(np.mean(test_rew))
            test_collision.append(np.mean(test_col))
            test_outRange.append(np.mean(test_out))
            test_done_steps.append(np.mean(test_done))

        # save the model
        if episode_gone > 0 and (episode_gone + 1) % arglist.fre_save_model == 0:
            time_now = time.strftime('%m%d_%H%M%S')
            print('=time:{} episode:{} step:{}        save'.format(time_now, episode_gone + 1, game_step))
            model_file_dir = os.path.join(save_policy_path, '{}_{}_{}'.format(
                arglist.scenario_name, time_now, (episode_gone + 1)))
            if not os.path.exists(model_file_dir):  # make the path
                os.mkdir(model_file_dir)
            for agent_idx, (a_c, a_t, c_c, c_t) in \
                    enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar)):
                torch.save(a_c, os.path.join(model_file_dir, 'a_c_{}.pt'.format(agent_idx)))
                torch.save(a_t, os.path.join(model_file_dir, 'a_t_{}.pt'.format(agent_idx)))
                torch.save(c_c, os.path.join(model_file_dir, 'c_c_{}.pt'.format(agent_idx)))
                torch.save(c_t, os.path.join(model_file_dir, 'c_t_{}.pt'.format(agent_idx)))

        episode_gone += 1

        if episode_gone > arglist.max_episode:
            # save the data
            train_rwd_file_name = save_plots_path + 'train_rewards.csv'
            np.savetxt(train_rwd_file_name, episode_rewards)

            evaluate_rew_file_name = save_plots_path + "test_rewards.csv"
            np.savetxt(evaluate_rew_file_name, test_reward)

            train_cov_file_name = save_plots_path + 'train_coverage_rate.csv'
            np.savetxt(train_cov_file_name, coverage_rate)

            evaluate_cov_file_name = save_plots_path + 'test_coverage_rate.csv'
            np.savetxt(evaluate_cov_file_name, test_coverage)

            collision_file_name = save_plots_path + "collision.csv"
            np.savetxt(collision_file_name, episode_collision)

            collision_file_name = save_plots_path + "test_collision.csv"
            np.savetxt(collision_file_name, test_collision)

            outRange_file_name = save_plots_path + "outRange.csv"
            np.savetxt(outRange_file_name, episode_outRange)

            outRange_file_name = save_plots_path + "test_outRange.csv"
            np.savetxt(outRange_file_name, test_outRange)

            modify_done_file_name = save_plots_path + "modify_done_steps.csv"
            np.savetxt(modify_done_file_name, modify_done_steps)

            done_steps_file_name = save_plots_path + 'test_done_steps.csv'
            np.savetxt(done_steps_file_name, test_done_steps)
            break


def create_dir():
    """构造目录 ./scenario_name/experiment_name/(plots, policy, buffer)"""
    scenario_path = "./" + scenario_name
    if not os.path.exists(scenario_path):
        os.mkdir(scenario_path)

    tm_struct = time.localtime(time.time())
    experiment_name = scenario_name + "_%02d_%02d_%02d_%02d" % \
                      (tm_struct[1], tm_struct[2], tm_struct[3], tm_struct[4])
    experiment_path = os.path.join(scenario_path, experiment_name)
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    save_paths = []
    save_paths.append(experiment_path + "/policy/")
    save_paths.append(experiment_path + "/plots/")
    for save_path in save_paths:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
    return save_paths[0], save_paths[1]


if __name__ == '__main__':
    save_policy_path, save_plots_path = create_dir()
    arg = parse_args()
    train(arg)
