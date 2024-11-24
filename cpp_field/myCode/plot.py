import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from matplotlib import rcParams

scenario_name = "coverage_0"

fileOrder = -1
data_path = "./" + scenario_name + "/" + os.listdir("./" + scenario_name)[fileOrder] + "/plots/"
figure_save_path = "./" + scenario_name + "/" + os.listdir("./" + scenario_name)[fileOrder] + "/figures/"

step = 100

# 设置图像字体
config = {
    "font.family": 'Times New Roman',  # 设置字体类型
}
rcParams.update(config)

if __name__ == '__main__':
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建

    reward = pd.read_csv(os.path.join(data_path, 'eva_rewards.csv'), sep=',', header=None, usecols=[0])
    coverage_rate = pd.read_csv(os.path.join(data_path, 'eva_coverage_rate.csv'), sep=',', header=None, usecols=[0])
    done_steps = pd.read_csv(os.path.join(data_path, 'done_steps.csv'), sep=',', header=None, usecols=[0])

    palette = plt.get_cmap('Set1')
    # 绘制奖励函数曲线
    plt.figure()
    plt.plot(reward, color=palette(1))
    plt.title("Evaluation Reward")
    plt.xlabel("episodes/%d" % step)
    plt.ylabel("reward")
    plt.grid()
    plt.savefig(os.path.join(figure_save_path, 'reward.jpg'), dpi=600)

    # 绘制覆盖率和连通率
    plt.figure()
    plt.plot(coverage_rate, color=palette(1))
    plt.title("Coverage Rate")
    plt.xlabel("episodes/%d" % step)
    plt.ylabel("Rate")
    plt.grid()
    plt.ylim([0.0, 1.1])
    plt.savefig(os.path.join(figure_save_path, 'Coverage_rate.jpg'), dpi=600)

    # 绘制done_step曲线
    plt.figure()
    plt.plot(done_steps, color=palette(1))
    plt.title("Done_steps")
    plt.xlabel("episodes/%d" % step)
    plt.ylabel("steps")
    plt.grid()
    plt.savefig(os.path.join(figure_save_path, 'done_steps.jpg'), dpi=600)