
import os

import numpy as np
from env.cpp_env.arguments_env_test import parse_args
import sys
import env.cpp_env.scenarios as scenarios
from env.cpp_env.environment import MultiAgentEnv

# sys.path.append(os.path.abspath("../multiagent_env/"))

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


def train(arglist):
    env = make_env(arglist.scenario_name, arglist)
    print('=============================')
    print('=1 Env {} is right ...'.format(arglist.scenario_name))
    print('=============================')

    """step2: create agents"""
    # n denotes agent numbers
    obs_shape_n = [env.observation_space[i].shape[0] for i in range(env.n)]
    action_shape_n = [env.action_space[i].n for i in range(env.n)]  # no need for stop bit

    print('=2 The {} agents are inited ...'.format(env.n))
    print('=============================')

    print('=3 starting iterations ...')
    print('=============================')
    while True:
        # get action  选择动作
        action_n = [np.random.rand(5) for _ in range(4)]
        # interact with env
        obs_n_t, rew_n, done_n = env.step(action_n)
        env.render()
        # if done_n:
        #     break


if __name__ == '__main__':
    arg = parse_args()
    train(arg)
