import torch


class Config:
    def __init__(self):
        self.train = True
        self.seed = 133
        self.cuda = True

        # train setting
        self.last_action = True  # 使用最新动作选择动作
        self.reuse_network = True  # 对所有智能体使用同一个网络
        # self.n_epochs = 100000  # 20000
        self.evaluate_epoch = 20  # 20
        self.evaluate_per_epoch = 100  # 100
        self.batch_size = 16  # 32
        self.buffer_size = int(5e3)
        self.save_frequency = 1000  # 训练多少次保存一次
        self.train_steps = 10  # 每次训练多少轮
        self.train_frequency = 5  # 多少场训练一次,上层训练频率
        self.episode_limit = 100  # 每个episode的最大步数，max_step
        self.max_episode = 10000  # 最大场数

        self.pretrain_episodes = 100  # 开始训练的场数
        self.fre_save_model = 500  # 上层网络模型保存的频率

        self.max_step = 140  # 一场最大步数
        self.high_net_step = 20  # 上层多少步算一步,注意self.max_step要整除self.high_net_step

        self.render_fra = 50  # render的场频率，多少场render一次
        self.render_step_fra = 5  # render的步频率，在一个render场中多少步render一次
        self.print_fre = 50  # 打印的频率

        self.epsilon_start = 0.5
        self.epsilon_end = 0.05
        self.epsilon_div = int(0.8 * self.max_episode)
        self.epsilon_step = (self.epsilon_start - self.epsilon_end) / float(self.epsilon_div)  # 每一轮贪婪系数下降值

        self.gamma = 0.99
        self.grad_norm_clip = 10  # prevent gradient explosion
        self.update_target_params = 20  # 200
        self.result_dir = './results/'

        # test setting
        self.load_model = False

        # SC2 env setting
        self.map_name = 'coverage_0'
        self.step_mul = 8  # 多少步执行一次动作
        self.difficulty = '2'
        self.game_version = 'latest'
        self.replay_dir = './replay_buffer/'
        self.field_mode = "gradual"  # 场强信息的显示形式，static为静态，gradual为动态

        if self.cuda:
            self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        print("CUDA",torch.cuda.is_available())

        # model structure
        self.n_actions = 2
        self.n_agents = 4
        # self.obs_shape = 113
        # self.state_shape = self.obs_shape * self.n_agents
        self.avail_action = [1] * self.n_actions

        # rnn net
        self.rnn_hidden_dim = 64
        # qmix net
        # input: (batch_size, n_agents, qmix_hidden_dim)
        self.qmix_hidden_dim = 32
        self.two_hyper_layers = False
        self.hyper_hidden_dim = 64
        self.model_dir = './models/'
        self.optimizer = "RMS"
        self.learning_rate = 5e-4

        # epsilon greedy
        self.start_epsilon = 1
        self.end_epsilon = 0.05
        self.anneal_steps = 50000  # 50000
        self.anneal_epsilon = (self.start_epsilon - self.end_epsilon) / self.anneal_steps
        self.epsilon_anneal_scale = 'step'

    def set_env_info(self, env_info):
        self.n_actions = env_info["n_actions"]
        self.state_shape = env_info["state_shape"]
        self.obs_shape = env_info["obs_shape"]
        self.n_agents = env_info["n_agents"]
        self.episode_limit = env_info["episode_limit"]
