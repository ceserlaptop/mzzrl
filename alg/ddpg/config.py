# 参数定义
import argparse  # 参数设置
import torch
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

# 创建解释器
parser = argparse.ArgumentParser()

# 参数定义
parser.add_argument('--actor_lr', type=float, default=3e-4, help='策略网络的学习率')
parser.add_argument('--critic_lr', type=float, default=3e-3, help='价值网络的学习率')
parser.add_argument('--n_hiddens', type=int, default=64, help='隐含层神经元个数')
parser.add_argument('--gamma', type=float, default=0.98, help='折扣因子')
parser.add_argument('--tau', type=float, default=0.005, help='软更新系数')
parser.add_argument('--buffer_size', type=int, default=10000, help='经验池容量')
parser.add_argument('--min_size', type=int, default=200, help='经验池超过200再训练')
parser.add_argument('--batch_size', type=int, default=64, help='每次训练64组样本')
parser.add_argument('--sigma', type=float, default=0.01, help='高斯噪声标准差')
parser.add_argument("--device", default=device, help="torch device ")

parser.add_argument('--action_bound', type=float, default=1.0, help='智能体动作的最大值')
parser.add_argument("--scenario_name", type=str, default="coverage_0", help="name of the scenario script")
parser.add_argument("--steps_per_train", type=int, default=10, help="下层网络搁多少个step_h训练一次")
parser.add_argument("--fre_save_model", type=int, default=500, help="网络模型保存的频率")
# 参数解析
args = parser.parse_args()
