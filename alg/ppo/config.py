import argparse


def get_args():
    parser = argparse.ArgumentParser(description="hyper parameters")

    parser.add_argument('--hidden_dim', default=256, type=int, help='hidden dim')
    parser.add_argument('--device', default='cpu', type=str, help="cpu or cuda")
    parser.add_argument('--actor_lr', default=0.0003, type=float, help="learning rate of actor net")
    parser.add_argument('--critic_lr', default=0.0003, type=float, help="learning rate of critic net")
    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser.add_argument('--gae_lambda', default=0.95, type=float, help='GAE lambda')
    parser.add_argument('--epochs', default=10, type=int, help='一条序列的数据用来训练轮数')
    parser.add_argument('--policy_clip', default=0.2, type=float, help='policy clip')

    args = parser.parse_args()
    return args
