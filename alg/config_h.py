import time
import argparse
import torch


class Config:
    def __init__(self):
        self.cuda = True
        self.scenario_name = "coverage_0"
        self.start_time = time.strftime('%y%m_%d%H%M')
        self.max_step = 160
        self.max_episode = 10000
        self.num_adversaries = 0  # 敌方的数量
        self.field_mode = "gradual"  # 场强信息的显示形式，static为静态，gradual为动态

        self.n_agent = 4
        self.n_skill = 2
        self.n_obs = 134
        self.render_fra = 10  # render的场频率，多少场render一次
        self.render_step_fra = 5  # render的步频率，在一个render场中多少步render一次
        self.high_net_step = 20  # 上层网络的步数间隔
        self.pretrain_episodes = 10  # 开始进行训练的场数
        self.steps_per_train = 100  # 下层网络搁多少个step_h训练一次
        self.epsilon_start = 0.5
        self.epsilon_end = 0.05
        self.epsilon_div = 1e3  # 贪婪系数下降场数

        self.print_fre = 500
        self.evaluate_episode_num = 10  # 一次评估的场数
        self.evaluate_episode_fre = 500  # 评估的频率，隔多少场评估一次
        self.save_cov_rate_threshold = 0.9  # 保存模型的阈值

        # checkpointing
        self.fre_save_model = 500  # 网络模型保存的频率
        self.save_dir = "models"  # 网络模型保存的位置

        # 网络参数
        self.low_level_alg = "ddpg"  # 目前使用的还是maddpg
        self.n_h1_low = 64
        self.n_h2_low = 64
        # self.high_level_alg = "qmix"  # 目前只有qmix
        # self.n_h1_high = 128
        # self.n_h2_high = 128
        # self.n_h_high_mixer = 64  # qmix混合网络层的神经元的数量
        # self.rnn_hidden_dim = 64  # rnn网络隐藏层神经元数量
        # self.policy_clip = 0.1  # ppo裁剪值
        if self.cuda:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        # self.safe_control = True  # 是否使用cbf安全控制器
        # self.learning_start_step = 10000  # learning start steps
        # self.max_grad_norm = 0.5  # max gradient norm for clip


def parse_args():
    parser = argparse.ArgumentParser("reinforcement learning experiments for multi-agent environments")

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
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")
    return parser.parse_args()
