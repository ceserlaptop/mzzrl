import time
import argparse

time_now = time.strftime('%y%m_%d%H%M')


def parse_args():
    parser = argparse.ArgumentParser("reinforcement learning experiments for multi-agent environments")
    # environment
    parser.add_argument("--scenario_name", type=str, default="coverage_0", help="name of the scenario script")
    parser.add_argument("--field_mode", type=str, default="gradual",
                        help="场强信息的显示形式，static为静态，gradual为动态")
    parser.add_argument("--agent_num", type=int, default=4, help="智能体数量")

    parser.add_argument("--benchmark", action="store_true", default=False)
    return parser.parse_args()
