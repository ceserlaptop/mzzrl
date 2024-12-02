# Programed by Z.Zhao
# 考虑连通保持的覆盖场景

import numpy as np
from env.cpp_env.CoverageWorld import CoverageWorld
from env.cpp_env.core import Agent, Landmark, Field_strength
from env.cpp_env.scenario import BaseScenario
from sklearn.cluster import KMeans


class Scenario(BaseScenario):
    def __init__(self, r_cover, r_comm, comm_r_scale, comm_force_scale, env_size, arglist):
        # agents的数量, 起飞位置, poi的数量和起飞位置
        self.num_agents = 4
        self.env_size = env_size
        self.agent_size = 0.02
        self.landmark_size = 0.02
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
        self.rew_done = 1500.0
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
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "poi_%d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = self.landmark_size
            landmark.m_energy = self.m_energy
            landmark.label = -1
        for i, field in enumerate(world.field_strength):
            field.name = "field_%d" % i

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

        x = np.arange(-0.9, 1, 0.2)
        y = np.arange(-0.9, 1, 0.2)
        X, Y = np.meshgrid(x, y)
        x = X.reshape(100, )
        y = Y.reshape(100, )
        pos_pois = np.array([x, y])
        if self.num_obst != 0:
            pos_pois = np.delete(pos_pois, self.mask_y + 10 * (9 - self.mask_x), axis=1)
        pos_pois = pos_pois.T
        for i, landmark in enumerate(world.landmarks):
            # landmark.color = np.array([1, 1, 1])
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.state.p_pos = pos_pois[i]
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.energy = 0.0
            landmark.done, landmark.just = False, False
            landmark.label = -1
        for i, field in enumerate(world.field_strength):
            field.state.p_pos = pos_pois[i]
            field.state.p_vel = np.zeros(world.dim_p)

        # self.area_partition(world)

        world.update_connect()

    def reward(self, agent, world, isreset):
        rew = 0.0
        count = 0
        # #判别下一时刻的位置是否与之前重复
        if isreset:
            self.agent_pos_tmp = self.agent_pos
        pos = agent.state.p_pos
        if pos in self.agent_pos_tmp:
            rew -= 1
        else:
            self.agent_pos_tmp = np.row_stack((self.agent_pos_tmp, pos))

        # cover new point and non-cover guide
        for i, poi in enumerate(world.landmarks):
            if not poi.done:
                for ag in world.agents:
                    if ag.target_id == poi.label:
                        # print("ag.target_id:", ag.target_id, "Poi.label:", poi.label)
                        rew -= np.linalg.norm(ag.state.p_pos - poi.state.p_pos)
                # 距离poi最近的uav, 二者之间的距离作为负奖励, 该poi的energy_to_cover为乘数

            elif poi.just:
                rew += self.rew_cover
                poi.just = False
                count += 1

        # stage reward
        rew += count / self.num_pois * self.rew_done
        # * (1 - train_ep / self.num_trainsp)
        # final reward
        if all([poi.done for poi in world.landmarks]):
            rew += self.rew_done
        # collision penalty
        for b, agent_b in enumerate(world.agents):
            for c, agent_c in enumerate(world.agents):
                if agent_c != agent_b:
                    da_tmp = np.linalg.norm(agent_c.state.p_pos - agent_b.state.p_pos)
                    if da_tmp < 0.35:
                        rew -= (0.35 - da_tmp) * 100
            # // + 0.15

            d_tmp = np.linalg.norm(agent_b.state.p_pos - np.array([0.3, 0.3]))
            if d_tmp < 0.35:
                rew -= (0.35 - d_tmp) * 100

            d1_tmp = np.linalg.norm(agent_b.state.p_pos - np.array([-0.6, 0.4]))
            # 0.53--0.45
            if d1_tmp < 0.3:
                rew -= (0.3 - d1_tmp) * 100
            d2_tmp = np.linalg.norm(agent_b.state.p_pos - np.array([-0.7, 0.7]))
            # 0.39--0.2
            if d2_tmp < 0.2:
                rew -= (0.2 - d2_tmp) * 100
            d3_tmp = np.linalg.norm(agent_b.state.p_pos - np.array([-0.7, 0.1]))
            # 0.39--0.2
            if d3_tmp < 0.2:
                rew -= (0.2 - d3_tmp) * 100
            d4_tmp = np.linalg.norm(agent_b.state.p_pos - np.array([-0.3, 0.5]))
            # 0.39--0.2
            if d4_tmp < 0.2:
                rew -= (0.2 - d4_tmp) * 100

            d5_tmp = np.linalg.norm(agent_b.state.p_pos - np.array([0.2, -0.4]))
            # 0.53--0.45
            if d5_tmp < 0.3:
                rew -= (0.3 - d5_tmp) * 100

            d6_tmp = np.linalg.norm(agent_b.state.p_pos - np.array([0.5, -0.5]))
            # 0.39--0.2
            if d6_tmp < 0.2:
                rew -= (0.2 - d6_tmp) * 100

            # out punish
            abs_pos = np.abs(agent_b.state.p_pos)
            # 0.9--0.95
            if abs_pos[0] > 0.95:
                rew -= (abs_pos[0] - 0.95) * 50
            # 0.9--0.95
            if abs_pos[1] > 0.95:
                rew -= (abs_pos[1] - 0.95) * 50
        return rew

    def observation(self, agent, world):
        info_agents = []
        for other in world.agents:
            if other is agent:
                continue
            info_agents.append(other.state.p_pos - agent.state.p_pos)
            info_agents.append([other.target_id])

        info_pois = []
        for poi in world.landmarks:
            info_pois.append(poi.state.p_pos - agent.state.p_pos)
            if poi.label == agent.target_id:
                info_pois.append([max(poi.m_energy - poi.energy, 0)])
            else:
                info_pois.append([0])
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [[agent.target_id]] + info_agents + info_pois)


    def done(self, agent, world):
        for ag in world.agents:
            abs_pos = np.abs(ag.state.p_pos)
            if (abs_pos > 1.2).any():
                return True
        return all([poi.done for poi in world.landmarks])
