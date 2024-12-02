import numpy as np
import matplotlib.pyplot as plt
from env.cpp_env import make_env
from env.cpp_env.arguments_env_test import parse_args
import torch
from alg.ppo.agent import PPO
import env.cpp_env.scenarios as scenarios
from env.cpp_env.environment import MultiAgentEnv

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


# ----------------------------------------- #
# 参数设置
# ----------------------------------------- #

num_episodes = 100  # 总迭代次数
gamma = 0.9  # 折扣因子
actor_lr = 1e-3  # 策略网络的学习率
critic_lr = 1e-2  # 价值网络的学习率
n_hiddens = 16  # 隐含层神经元个数

return_list = []  # 保存每个回合的return

# ----------------------------------------- #
# 环境加载
# ----------------------------------------- #
arg = parse_args()
env = make_env(arg)
n_states = 314
n_actions = 5  # no need for stop bit

# ----------------------------------------- #
# 模型构建
# ----------------------------------------- #
agent = PPO(n_states=n_states,  # 状态数
            n_hiddens=n_hiddens,  # 隐含层数
            n_actions=n_actions,  # 动作数
            actor_lr=actor_lr,  # 策略网络学习率
            critic_lr=critic_lr,  # 价值网络学习率
            lmbda=0.95,  # 优势函数的缩放因子
            epochs=10,  # 一组序列训练的轮次
            eps=0.2,  # PPO中截断范围的参数
            gamma=gamma,  # 折扣因子
            device=device
            )

# ----------------------------------------- #
# 训练--回合更新 on_policy
# ----------------------------------------- #

for i in range(num_episodes):
    state = env.reset()[0]  # 环境重置
    done = False  # 任务完成的标记
    episode_return = 0  # 累计每回合的reward

    # 构造数据集，保存每个回合的状态数据
    transition_dict = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': [],
    }

    while not done:
        action = agent.take_action(state[0])  # 动作选择
        next_state, reward, done, _, _ = env.step(action)  # 环境更新
        # 保存每个时刻的状态\动作\...
        transition_dict['states'].append(state)
        transition_dict['actions'].append(action)
        transition_dict['next_states'].append(next_state)
        transition_dict['rewards'].append(reward)
        transition_dict['dones'].append(done)
        # 更新状态
        state = next_state
        # 累计回合奖励
        episode_return += reward
        env.render()

    # 保存每个回合的return
    return_list.append(episode_return)
    # 模型训练
    agent.learn(transition_dict)

    # 打印回合信息
    print(f'iter:{i}, return:{np.mean(return_list[-10:])}')

# -------------------------------------- #
# 绘图
# -------------------------------------- #

plt.plot(return_list)
plt.title('return')
plt.show()
