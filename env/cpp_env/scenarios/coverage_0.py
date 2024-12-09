# Programed by Z.Zhao
# 考虑连通保持的覆盖场景

import numpy as np
import pandas as pd

from env.cpp_env.CoverageWorld import CoverageWorld
from env.cpp_env.core import Agent, Landmark, Field_strength, Grid
from env.cpp_env.scenario import BaseScenario
from alg.utils import choose_point
from alg.utils import calculate_pos


class Scenario(BaseScenario):
    def __init__(self, r_cover, r_comm, comm_r_scale, comm_force_scale, env_size, arglist):
        # agents的数量, 起飞位置, poi的数量和起飞位置
        self.num_agents = 4
        self.env_size = env_size
        self.grid_size = 5  # 简化后的大格子地图为几乘几：5*5
        self.agent_size = 0.02
        self.landmark_size = 0.02
        self.scope = 1  # 表示场地范围，取值为1表示范围为x:[-1,1];y:[-1,1]
        # 障碍物所占位数量
        self.num_obst = 0
        self.num_pois = self.env_size * self.env_size - self.num_obst
        # self.done_poi = []  # 用来存储已经完成覆盖的目标点
        # self.pos_agents = [[[x, x], [x, x], [x, x], [x, x]] for x in [0.9]][0]
        # 智能体初始位置
        self.agent_pos = np.array([[-0.5, -0.5], [-0.1, -0.5], [-0.5, -0.1], [-0.1, -0.1]])
        self.agent_pos_tmp = self.agent_pos
        self.mask_x = np.array([])  # 保存障碍物栅格x坐标
        self.mask_y = np.array([])  # 保存障碍物栅格y坐标
        self.r_cover = r_cover
        self.r_comm = r_comm
        self.m_energy = 1.0

        self.rew_cover = 20.0
        self.rew_field = 5.0  # 场强上升一个单位的奖励

        self.rew_done = 1500.0
        self.rew_out = -100.0
        self.rew_collision = -20.0
        self.rew_unconnect = -10.0
        self.num_trainsp = 100000
        self.comm_r_scale = comm_r_scale  # r_comm * comm_force_scale = 计算聚合力时的通信半径
        self.comm_force_scale = comm_force_scale  # 连通保持聚合力的倍数
        self.occupy_map = np.zeros((self.env_size, self.env_size))
        self.arglist = arglist

    def make_world(self):
        if self.num_obst != 0:
            mask = np.array(
                [[0.1, 0.1], [0.1, 0.3], [0.3, 0.3], [0.1, 0.5], [0.5, 0.3], [0.3, 0.1], [0.3, 0.5], [0.5, 0.5],
                 [0.5, 0.1], [0.1, -0.3], [0.3, -0.3], [0.1, -0.5], [0.3, -0.5], [0.5, -0.5], [-0.7, 0.7],
                 [-0.7, 0.5], [-0.5, 0.5], [-0.3, 0.5], [-0.7, 0.3], [-0.5, 0.3], [-0.7, 0.1]])
            self.mask_x = 10 - np.around((mask[:, 1] + 1.1) / 0.20)
            self.mask_y = np.around((mask[:, 0] + 1.1) / 0.2) - 1
            self.mask_x = self.mask_x.astype(int)
            self.mask_y = self.mask_y.astype(int)
            self.occupy_map[self.mask_x, self.mask_y] = 1
        cover_map = self.occupy_map

        world = CoverageWorld(self.comm_r_scale, self.comm_force_scale)
        world.collaborative = True
        world.env_size = self.env_size
        world.arglist = self.arglist

        world.agents = [Agent() for _ in range(self.num_agents)]  # 代表UAV, size为覆盖面积
        world.landmarks = [Landmark() for _ in range(self.num_pois)]
        world.field_strength = [Field_strength() for _ in range(self.num_pois)]
        world.grid = [Grid() for _ in range(self.grid_size * self.grid_size)]
        # 定义场强，后期要初始化为真实的场强数据
        # world.field_strength = [50] * int(self.num_pois / 3) + [100] * int(self.num_pois / 3) + [75] * (
        #             self.num_pois - int(self.num_pois / 3) * 2)

        for i, agent in enumerate(world.agents):
            agent.name = "agent_%d" % i
            agent.collide = False
            agent.silent = True
            agent.size = self.agent_size
            agent.r_cover = self.r_cover
            agent.r_comm = self.r_comm
            agent.max_speed = 1.0
            agent.target_id = -1

        # 真实的场强数据
        data_path = "gradual_elliptical_field_strength.csv"
        data = pd.read_csv(data_path)
        data_values = data.values.reshape(100)

        pos_pois = calculate_pos(self.env_size, 1)
        if self.num_obst != 0:
            pos_pois = np.delete(pos_pois, self.mask_y + 10 * (9 - self.mask_x), axis=1)
        pos_pois = pos_pois.T

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = pos_pois[i]
            landmark.field_really = data_values[i]  # 给每一个目标点进行赋值
            landmark.name = "poi_%d" % i  # 目标点名字
            landmark.collide = False
            landmark.movable = False
            landmark.size = self.landmark_size
            landmark.m_energy = self.m_energy
            landmark.label = -1
        for i, field in enumerate(world.field_strength):
            field.state.p_pos = pos_pois[i]
            field.name = "field_%d" % i

        pos_grid = calculate_pos(self.grid_size, self.scope)
        pos_grid = pos_grid.T
        little_l = self.scope / world.env_size
        grid_l = 1.5 * (world.env_size / self.grid_size - 1) * little_l
        for i, grid in enumerate(world.grid):
            grid.name = "grid_%d" % i
            grid.state.p_pos = pos_grid[i]
            grid.sub_points = []
            for landmark in world.landmarks:
                distance = landmark.state.p_pos - grid.state.p_pos
                if abs(distance[0]) <= grid_l and abs(distance[1]) <= grid_l:
                    grid.sub_points.append(landmark)
                    pass
        self.reset_world(world)

        return world

    def reset_world(self, world):
        # world.done_poi = []
        for i, agent in enumerate(world.agents):
            # agent.color = np.array([0.05, 0.15, 0.05])
            agent.color = np.array([0.7, 0.7, 0.7])
            agent.cover_color = np.array([0.5, 0.5, 0.5])
            # agent.cover_color = np.array([0.05, 0.25, 0.05])
            agent.comm_color = np.array([0.05, 0.35, 0.05])
            agent.state.p_pos = np.array(self.agent_pos[i])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.target_id = -1

        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.energy = 0.0
            landmark.done, landmark.just = False, False
            landmark.label = -1

        for i, field in enumerate(world.field_strength):
            field.state.p_vel = np.zeros(world.dim_p)

        for i, grid in enumerate(world.grid):
            grid.done, grid.just = False, False
            grid.repeat_time = 0
        # self.area_partition(world)
        # 初始化采到的场强数据列表，形式：编号和数值
        world.get_field_data = []
        world.update_connect()

    def reward(self, agent, world, action_h):
        rew = 0.0
        if action_h == 0:  # 覆盖模式
            for grid in agent.cov_grid:  # 大网格中的目标点
                if grid.just:  # 新覆盖的大网格
                    grid.just = False  # 只能获取一次奖励
                    rew += self.rew_cover * len(grid.sub_points)
            agent.cov_grid = []  # 将一次覆盖的grid清0

        elif action_h == 1:  # 高场强模式
            if len(agent.get_field) > 0:
                average_field = sum(agent.get_field) / len(agent.get_field)
                agent.get_field = []  # 清空智能体每一步采的场强点
            else:  # 没有覆盖新的点
                average_field = agent.last_field
            if agent.last_field is None:  # 初始的场强，直接给当前值
                agent.last_field = average_field
            rew = (average_field - agent.last_field) * self.rew_field

            # if average_field != agent.last_field:
            #     print(agent.name, average_field, agent.last_field)

            agent.last_field = average_field  # 进行记录
        else:
            print("coverage_0.py中的reward函数报错，无此高层动作，请检查代码")
        # 出界惩罚
        abs_pos = np.abs(agent.state.p_pos)
        rew += np.sum(abs_pos[abs_pos > 1] - 1) * self.rew_out  # 对出界部分的长度进行计算
        if (abs_pos > 1.2).any():  # 出界太多，则直接给惩罚
            rew += self.rew_out
        # 相互碰撞惩罚
        # for i, ag in enumerate(world.agents):
        for other_agent in world.agents:
            if other_agent.name != agent.name:
                dist = np.linalg.norm(agent.state.p_pos - other_agent.state.p_pos)
                if dist < 0.2:
                    rew += self.rew_collision
        return rew

    def observation(self, agent, world):
        info_agents = []
        for other in world.agents:
            if other is agent:
                continue
            info_agents.append(other.state.p_pos - agent.state.p_pos)
            info_agents.append([other.target_id])

        info_grid = []
        for grid in world.grid:
            dis = grid.state.p_pos - agent.state.p_pos
            done = grid.done
            info_grid.append([*dis, done])

        # 加入四个方向的场强信息，固定为四个方向最近的8个点
        field_list = choose_point(agent, world.field_strength)
        info_field = []
        for field in field_list:
            for i in range(len(field)):
                if field[i][0] is None:  # 如果这个方向没有目标点，则直接给场强为0，坐标为智能体的坐标
                    field[i][1] = 0
                    field[i][0] = agent.state.p_pos
                else:
                    field[i][1] = field[i][0].field_data  # 给到估计的场强
                    field[i][0] = field[i][0].state.p_pos  # 给到坐标点
                info_field.append([*field[i][0], field[i][1]])
        return np.concatenate(
            [agent.state.p_vel] + [agent.state.p_pos] + [[agent.target_id]] + info_agents + info_grid + info_field)

    def done(self, agent, world):
        for ag in world.agents:
            abs_pos = np.abs(ag.state.p_pos)
            if (abs_pos > 1.2).any():
                return True
        return all([poi.done for poi in world.landmarks])
