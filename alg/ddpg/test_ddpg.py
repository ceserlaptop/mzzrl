import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
from alg.ddpg.config import args
from alg.ddpg.agent import ReplayBuffer, DDPG
#
# print('CUDA版本:', torch.version.cuda)
# print('Pytorch版本:', torch.__version__)
# print('显卡是否可用:', '可用' if (torch.cuda.is_available()) else '不可用')
# print('显卡数量:', torch.cuda.device_count())
# print('是否支持BF16数字格式:', '支持' if (torch.cuda.is_bf16_supported()) else '不支持')
# print('当前显卡型号:', torch.cuda.get_device_name())
# print('当前显卡的CUDA算力:', torch.cuda.get_device_capability())
# print('当前显卡的总显存:', torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024, 'GB')
# print('是否支持TensorCore:', '支持' if (torch.cuda.get_device_properties(0).major >= 7) else '不支持')
# print('当前显卡的显存使用率:', torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory * 100,
#       '%')

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

# -------------------------------------- #
# 环境加载
# -------------------------------------- #

env_name = "MountainCarContinuous-v0"  # 连续型动作
env = gym.make(env_name)
n_states = env.observation_space.shape[0]  # 状态数 2
n_actions = env.action_space.shape[0]  # 动作数 1
action_bound = env.action_space.high[0]  # 动作的最大值 1.0
render_interval = 25
# -------------------------------------- #
# 模型构建
# -------------------------------------- #

# 经验回放池实例化
replay_buffer = ReplayBuffer(capacity=args.buffer_size)
# 模型实例化
agent = DDPG(n_states=n_states,  # 状态数
             n_actions=n_actions,  # 动作数
             action_bound=action_bound,  # 动作最大值
             args=args)

# -------------------------------------- #
# 模型训练
# -------------------------------------- #

return_list = []  # 记录每个回合的return
mean_return_list = []  # 记录每个回合的return均值

for i in range(1000):  # 迭代10回合
    episode_return = 0  # 累计每条链上的reward
    state = env.reset()  # 初始时的状态
    done = False  # 回合结束标记

    while not done:
        # 获取当前状态对应的动作
        action = agent.take_action(state)
        # 环境更新
        next_state, reward, done, _ = env.step(action)

        reward = abs(next_state[1])*0.1
        if next_state[1] > 0:  # 向右运动
            reward = reward + 0.1
        # else:
        #     reward += -0.1
        # if next_state[0] >= 0.5:
        #     reward += 10

        # 更新经验回放池
        replay_buffer.add(state, action, reward, next_state, done)
        # 状态更新
        state = next_state
        # 累计每一步的reward
        episode_return += reward
        if i % render_interval == 0:
            env.render()

        # 如果经验池超过容量，开始训练
        if replay_buffer.size() > args.min_size:
            # 经验池随机采样batch_size组
            s, a, r, ns, d = replay_buffer.sample(args.batch_size)
            # 构造数据集
            transition_dict = {
                'states': s,
                'actions': a,
                'rewards': r,
                'next_states': ns,
                'dones': d,
            }
            # 模型训练
            agent.update(transition_dict)

    # 保存每一个回合的回报
    return_list.append(episode_return)
    mean_return_list.append(np.mean(return_list[-100:]))  # 平滑

    # 打印回合信息
    print(f'iter:{i}, return:{episode_return}, mean_return:{np.mean(return_list[-10:])}')

# 关闭动画窗格
env.close()

# -------------------------------------- #
# 绘图
# -------------------------------------- #

x_range = list(range(len(return_list)))

plt.subplot(121)
plt.plot(x_range, return_list)  # 每个回合return
plt.xlabel('episode')
plt.ylabel('return')
plt.subplot(122)
plt.plot(x_range, mean_return_list)  # 每回合return均值
plt.xlabel('episode')
plt.ylabel('mean_return')
