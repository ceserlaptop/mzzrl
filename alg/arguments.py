# Time: 2019-11-05
# Author: Zachary 
# Name: MADDPG_torch

import time
import argparse
import torch

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
time_now = time.strftime('%y%m_%d%H%M')


def parse_args():
    parser = argparse.ArgumentParser("reinforcement learning experiments for multi-agent environments")

    # environment
    parser.add_argument("--scenario_name", type=str, default="coverage_0", help="name of the scenario script")
    parser.add_argument("--start_time", type=str, default=time_now, help="the time when start the game")
    parser.add_argument("--max_step", type=int, default=160, help="maximum episode length")
    parser.add_argument("--max_episode", type=int, default=10000, help="maximum episode length")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--field_mode", type=str, default="gradual", help="场强信息的显示形式，static为静态，gradual为动态")
    parser.add_argument("--n_agent", type=int, default=4, help="智能体数量")
    parser.add_argument("--n_skill", type=int, default=2, help="上层技巧的数量")
    parser.add_argument("--n_obs", type=int, default=134, help="观测数据的维度")
    parser.add_argument("--render_fra", type=int, default=10, help="render的频率")
    parser.add_argument("--render_step_fra", type=int, default=5, help="render时多少步render一次")

    # 训练相关参数设置
    parser.add_argument("--high_net_step", type=int, default=20, help="上层网络的步数间隔")
    parser.add_argument("--traj_skip", type=int, default=4, help="表示输入的一段轨迹被分成几段")
    parser.add_argument("--use_state_difference", type=bool, default=True, help="如果为真，则使用观测值之间的差异，而不是原始观测值本身，作为解码器的轨迹输入")
    parser.add_argument("--obs_truncate_length", type=int, default=10, help="编码器输入的观测数据在每个时间步的长度")

    parser.add_argument("--pretrain_episodes", type=int, default=10, help="开始进行训练的场数")
    parser.add_argument("--steps_per_train", type=int, default=10, help="下层网络搁多少个step_h训练一次")
    parser.add_argument("--epsilon_start", type=float, default=0.5, help="初始的贪婪系数")
    parser.add_argument("--epsilon_end", type=float, default=0.05, help="最终的贪婪系数")
    parser.add_argument("--epsilon_div", type=float, default=1e3, help="贪婪系数下降场数")

    parser.add_argument("--alpha_start", type=float, default=1.0, help="初始的权重系数")
    parser.add_argument("--alpha_end", type=float, default=0.6, help="最终的权重系数")
    parser.add_argument("--alpha_step", type=float, default=0.05, help="下降的权重系数")
    parser.add_argument("--alpha_threshold", type=float, default=0.5, help="权重系数阈值")

    parser.add_argument("--decoder_dataset_train", type=float, default=0.5, help="开始训练编码器的数据集大小")
    parser.add_argument("--print_fre", type=int, default=500, help="打印的场数间隔")
    parser.add_argument("--evaluate_episode_num", type=int, default=10, help="一次评估的场数")
    parser.add_argument("--evaluate_episode_fre", type=int, default=500, help="评估的频率，搁多少场评估一次")
    parser.add_argument("--save_cov_rate_threshold", type=int, default=0.9, help="保存模型的阈值")

    # checkpointing
    parser.add_argument("--fre_save_model", type=int, default=500, help="网络模型保存的频率")
    parser.add_argument("--save_dir", type=str, default="models", help="网络模型保存的位置")

    # 网络参数
    parser.add_argument("--low_level_alg", type=str, default="ppo", help="下层网络的类型,目前只有ppo")
    parser.add_argument("--n_h1_low", type=int, default=64, help="下层网络第一层隐藏层的神经元的数量")
    parser.add_argument("--n_h2_low", type=int, default=64, help="下层网络第二层隐藏层的神经元的数量")
    parser.add_argument("--high_level_alg", type=str, default="qmix", help="下层网络的类型,目前只有qmix")
    parser.add_argument("--n_h1_high", type=int, default=128, help="上层网络第一层隐藏层的神经元的数量")
    parser.add_argument("--n_h2_high", type=int, default=128, help="上层网络第二层隐藏层的神经元的数量")
    parser.add_argument("--n_h_high_mixer", type=int, default=64, help="qmix混合网络层的神经元的数量")
    parser.add_argument("--decode_hidden_size", type=int, default=64, help="编码器的隐藏层神经元数量")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="rnn网络隐藏层神经元数量")
    parser.add_argument("--policy_clip", type=float, default=0.1, help="ppo裁剪值")

    parser.add_argument("--device", default=device, help="torch device ")
    parser.add_argument("--safe_control", type=bool, default=True, help="adopt the CBF ")
    parser.add_argument("--learning_start_step", type=int, default=10000, help="learning start steps")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="max gradient norm for clip")
    parser.add_argument("--learning_fre", type=int, default=100, help="learning frequency")
    parser.add_argument("--tau", type=int, default=0.01, help="how depth we exchange the par of the nn")
    parser.add_argument("--lr_a", type=float, default=0.002, help="learning rate for adam optimizer")
    parser.add_argument("--lr_c", type=float, default=0.002, help="learning rate for adam optimizer")
    parser.add_argument("--lr_qmix", type=float, default=0.002, help="learning rate for adam optimizer")
    parser.add_argument("--lr_ppo", type=float, default=0.002, help="learning rate for adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch_size", type=int, default=2, help="number of episodes to optimize at the same time")
    parser.add_argument("--buffer_size", type=int, default=10, help="number of data stored in the memory")
    parser.add_argument("--num_units_1", type=int, default=128, help="number of units in the mlp")
    parser.add_argument("--num_units_2", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--num_units_openai", type=int, default=128, help="number of units in the mlp")

    # evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()
