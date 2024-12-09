import numpy as np
from env.cpp_env.core import World
from scipy.optimize import minimize
import pandas as pd
from alg.utils import get_rgb


class CoverageWorld(World):
    def __init__(self, comm_r_scale=0.9, comm_force_scale=0.0):
        super(CoverageWorld, self).__init__()
        self.coverage_rate = 0.0  # 每次step后重新计算
        self.connect = False  # 当前是否强连通
        self.dist_mat = np.zeros([4, 4])  # agents之间的距离矩阵, 对角线为1e5
        # self.adj_mat = np.zeros([4, 4])  # 标准r_comm下的邻接矩阵, 对角线为0
        # self.damping = 0.25

        # 对连通保持聚合力的修正
        self.contact_force *= comm_force_scale  # 修正拉力倍数
        self.comm_r_scale = comm_r_scale  # 产生拉力的半径 = r_comm * comm_r_scale

        # 在comm_r_scale修正下的强连通和邻接矩阵
        self.connect_ = False  # 用于施加规则拉力的强连通指示
        self.adj_mat_ = np.zeros([4, 4])  # 用于施加规则拉力的邻接矩阵

        self.dt = 0.1

        # 添加障碍物位置信息
        self.obst_pos = []
        self.obst_pos.append(np.array([0.1, 0.5, 0.1, 0.5]))
        self.arglist = None
        self.get_field_data = []

    def revise_action(self, action_n):
        action_r = np.zeros_like(action_n)
        p_force = [np.zeros(self.dim_p) for _ in self.agents]
        sensitivity = 2.0

        for i in range(len(self.agents)):
            p_force[i] += [action_n[i][1] - action_n[i][2], action_n[i][3] - action_n[i][4]]

        if self.contact_force > 0:
            p_force = self.apply_connect_force(p_force)

        for action, force in zip(action_r, p_force):
            action[0] = 0
            for i in range(2):
                if force[i] > 0:
                    action[2 * i + 1] = force[i]
                    action[2 * i + 2] = 0
                elif force[i] < 0:
                    action[2 * i + 1] = 0
                    action[2 * i + 2] = -force[i]
            action /= sensitivity

        for i in range(4):
            action_r[i] = np.exp(action_r[i]) / np.sum(np.exp(action_r[i]))
        return action_r

    def step(self, action_n):
        # step函数中, 不计算poi受力与位移, 增加保持连通所需的拉力, 规则如下:
        # self.update_connect()  # 获得adj_mat(_), dist_mat, connect(_)
        p_velx = []
        for i, agent in enumerate(self.agents):
            vel = []
            vel.append(agent.state.p_vel)
            p_velx.append(vel)
        p_force = [None for _ in range(len(self.agents))]
        p_force = self.apply_action_force(p_force)

        self.integrate_state(p_force)  # 更新状态
        self.update_energy()  # 更新能量
        self.field_change(self.arglist.field_mode)  # 更新估计的场强

    def update_connect(self):
        # 更新邻接矩阵adj_mat和adj_mat_, adj对角线为0, dist对角线为1e5
        self.adj_mat = np.zeros([len(self.agents), len(self.agents)])
        self.adj_mat_ = np.zeros([len(self.agents), len(self.agents)])
        for a, agent_a in enumerate(self.agents):
            for b, agent_b in enumerate(self.agents):
                self.dist_mat[a, b] = np.linalg.norm(agent_a.state.p_pos - agent_b.state.p_pos)
                if self.dist_mat[a, b] < agent_a.r_comm + agent_b.r_comm:
                    self.adj_mat[a, b] = 1
                    if self.dist_mat[a, b] < self.comm_r_scale * (agent_a.r_comm + agent_b.r_comm):
                        self.adj_mat_[a, b] = 1
            self.dist_mat[a, a] = 1e5
            self.adj_mat[a, a] = 0
            self.adj_mat_[a, a] = 0

        # 更新connect和connect_
        connect_mat = [np.eye(len(self.agents))]
        connect_mat_ = [np.eye(len(self.agents))]
        for _ in range(len(self.agents) - 1):
            connect_mat.append(np.matmul(connect_mat[-1], self.adj_mat))
            connect_mat_.append(np.matmul(connect_mat[-1], self.adj_mat_))

        self.connect = True if (sum(connect_mat) > 0).all() else False
        self.connect_ = True if (sum(connect_mat_) > 0).all() else False

    def apply_action_force(self, p_force):
        for i, agent in enumerate(self.agents):
            p_force[i] = agent.action.u
        return p_force

    def apply_connect_force(self, p_force):
        # 对强连通分支A, 计算其与所有其他强连通分支的距离, 取最短距离, 在此距离的两端产生拉力,
        # 4agent的简化版本:
        # 1) 对没有和其他agent建立连接的孤立agent, 会受到与他最近的agent之间的拉力
        # 2) 若所有agent均有邻居但未达到全连接, 则对当前所有距离里比通信距离大的最小距离添加拉力
        if self.connect_:
            return p_force
        tmp_mat = np.sum(self.adj_mat_, 0)  # 列和为0的agent表示没有连接
        idxs = np.where(tmp_mat == 0)[0]  # idxs 为孤立agent的索引
        # 1) 对孤立agent, 受到与它最近的agent之间的拉力
        if len(idxs) > 0:
            for a in idxs:
                dists = self.dist_mat[a, :]  # 孤立agent与其他agent的距离(与自己的为1e5)
                b = np.argmin(dists)  # 距离孤立agent最近的agent的索引
                [f_a, f_b] = self.get_connect_force(self.agents[a], self.agents[b])
                p_force[a] += f_a
                p_force[b] += f_b
        # 2) 若所有agent均有邻居但未达到全连接, 则对当前所有距离里比通信距离大的最小距离添加拉力
        else:
            idx1 = (self.dist_mat < self.comm_r_scale * 2 * self.agents[0].r_comm)
            self.dist_mat[idx1] = 1e5
            idx2 = np.argmin(self.dist_mat)
            a = idx2 // len(self.agents)
            b = idx2 % len(self.agents)
            [f_a, f_b] = self.get_connect_force(self.agents[a], self.agents[b])
            p_force[a] += f_a
            p_force[b] += f_b
        return p_force

    def get_connect_force(self, agent_a, agent_b):
        if agent_a is agent_b:
            return [0, 0]
        delta_pos = agent_a.state.p_pos - agent_b.state.p_pos
        dist = np.linalg.norm(delta_pos)
        dist_max = (agent_a.r_comm + agent_b.r_comm) * self.comm_r_scale
        k = self.contact_margin
        penetration = np.logaddexp(0, (dist - dist_max) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = -force
        force_b = +force
        return [force_a, force_b]

    def integrate_state(self, p_force):
        for i, entity in enumerate(self.agents):
            if not entity.movable:
                continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt

            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                      np.square(
                                                                          entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

    def check_state(self, p_force):
        # 将p_force输入给系统，先测试一下是否会失去连通
        tmp_vel = [ag.state.p_vel * 1 for ag in self.agents]
        tmp_pos = [ag.state.p_pos * 1 for ag in self.agents]
        for i, agent in enumerate(self.agents):
            tmp_vel[i] = agent.state.p_vel * (1 - self.damping)
            if p_force[i] is not None:
                tmp_vel += (p_force[i] / agent.mass) * self.dt
            if agent.max_speed is not None:
                speed = np.linalg.norm(tmp_vel[i])
                if speed > agent.max_speed:
                    tmp_vel[i] = tmp_vel[i] / speed * agent.max_speed
            tmp_pos[i] += tmp_vel[i] * self.dt
        return tmp_pos

    def check_connect(self, tmp_pos):
        # 更新邻接矩阵adj_mat和adj_mat_, adj对角线为0, dist对角线为1e5
        self.adj_mat = np.zeros([len(self.agents), len(self.agents)])
        for a, agent_a in enumerate(self.agents):
            for b, agent_b in enumerate(self.agents):
                self.dist_mat[a, b] = np.linalg.norm(tmp_pos[a] - tmp_pos[b])
                if self.dist_mat[a, b] < agent_a.r_comm + agent_b.r_comm:
                    self.adj_mat[a, b] = 1
            self.dist_mat[a, a] = 1e5
            self.adj_mat[a, a] = 0
        # 更新connect
        connect_mat = [np.eye(len(self.agents))]
        for _ in range(len(self.agents) - 1):
            connect_mat.append(np.matmul(connect_mat[-1], self.adj_mat))
        return True if (sum(connect_mat) > 0).all() else False

    def update_energy(self):
        num_done = 0
        for poi in self.landmarks:
            if poi.done:
                num_done += 1
            else:
                for agent in self.agents:
                    dist = np.linalg.norm(poi.state.p_pos - agent.state.p_pos)
                    if dist <= agent.r_cover:
                        # poi.energy += (1 - dist / agent.r_cover)  # power随半径线性减少
                        poi.energy += 1
                        agent.get_field.append(poi.field_really)
                if poi.energy >= poi.m_energy:
                    # self.done_poi.append(poi)
                    poi.done = True
                    poi.just = True
                    for grid in self.grid:
                        if (not grid.done) and (poi in grid.sub_points):
                            grid.done = True
                            poi.just = True
                    self.get_field_data.append([poi.name, poi.field_really])  # 加入智能体观测到的真实数据的列表
                    print("已采集数据", self.get_field_data)
                    num_done += 1
                poi.color = np.array([0.25 + poi.energy / poi.m_energy * 0.75, 0.25, 0.25])
        self.coverage_rate = num_done / len(self.landmarks)

    def field_change(self, mode):
        file_path = "gradual_elliptical_field_strength.csv"  # 场强数据文件路径
        data_path = "gradual_elliptical_field_strength.csv"
        data = pd.read_csv(data_path)
        data_values = data.values.reshape(100)
        rgb_data = get_rgb(file_path)  # 读取并转为 NumPy 数组
        i = 0
        for field in self.field_strength:
            field.field_data = data_values[i]
            field_name_num = (field.name.split("_")[-1])  # 这里得到的是字符串型的数据
            if mode == "static":
                # 全局不变
                poi_field = None
                # 取出对应的poi
                for poi in self.landmarks:
                    if poi.name == "poi_" + (field.name.split("_")[-1]):
                        poi_field = poi
                # 判断对应的点是否完成覆盖
                if poi_field.done:
                    field.color = rgb_data[int(field_name_num)]
                else:
                    field.color = np.array([1, 1, 1])
            elif mode == "gradual":
                # 全局动态变化
                field.color = rgb_data[int(field_name_num)]
            else:
                print("场强显示模式错误")
            i += 1

    def apply_outrange_force(self, tmp_pos, p_force):
        # 判断是否出界, 并施加拉力
        for id, pos in enumerate(tmp_pos):
            abs_pos = np.abs(pos)
            if (abs_pos > 1).any():
                x = pos[0]
                y = pos[1]
                c = 1
                if x - 1 > 0:  # 從右邊出界
                    d = x - 1
                    aa = c * ((d - self.agents[id].r_cover) / (d - self.agents[id].size)) ** 2
                    p_force[id] += 0.2 * aa * np.array([-1, 0])
                elif x + 1 < 0:  # 從左邊出界
                    d = -1 - x
                    aa = c * ((d - self.agents[id].r_cover) / (d - self.agents[id].size)) ** 2
                    p_force[id] += 0.6 * aa * np.array([1, 0])
                elif y - 1 > 0:  # 從上面出界
                    d = y - 1
                    aa = c * ((d - self.agents[id].r_cover) / (d - self.agents[id].size)) ** 2
                    p_force[id] += 0.6 * aa * np.array([0, -1])
                else:  # 從下面出界
                    d = -1 - y
                    aa = c * ((d - self.agents[id].r_cover) / (d - self.agents[id].size)) ** 2
                    p_force[id] += 0.2 * aa * np.array([0, 1])

        return p_force

    # 10 * 10

    def func(self, args):
        #
        fu_k, fd_k, fr_k, fl_k = args
        v = lambda x: (x[0] - fu_k) ** 2 + (x[1] - fd_k) ** 2 + (x[2] - fr_k) ** 2 + (x[3] - fl_k) ** 2
        return v

    def con_up(self, args):
        m, ddt, alfa, U_b, py, vy, yita = args
        cons = ({'type': 'ineq',
                 # 'fun': lambda x: m / ddt * (alfa * (U_b - py - vy * ddt) - vy) - m * (1 - yita) * vy - x[0] + x[1]})
                 'fun': lambda x: - vy - (1 - yita) * vy * ddt - ddt / m * (x[0] - x[1]) + alfa * (
                         U_b - py - vy * ddt)})
        return cons

    def con_down(self, args):
        m, ddt, alfa, D_b, py, vy, yita = args
        cons = ({'type': 'ineq',
                 # 'fun': lambda x: m * (1 - yita) * vy - m / ddt * (alfa * (D_b - py - vy * ddt) - vy) + x[0] - x[1]})
                 'fun': lambda x: vy + (1 - yita) * vy * ddt + ddt / m * (x[0] - x[1]) + alfa * (py + vy * ddt - D_b)})
        return cons

    def con_left(self, args):
        m, ddt, alfa, L_b, px, vx, yita = args
        cons = ({'type': 'ineq',
                 # 'fun': lambda x: x[2] - x[3] + m * (1 - yita) * vx - m / ddt * (alfa * (L_b - px - vx * ddt) - vx)})
                 'fun': lambda x: vx + (1 - yita) * vx * ddt + ddt / m * (x[2] - x[3]) + alfa * (px + vx * ddt - L_b)})
        return cons

    def con_right(self, args):
        m, ddt, alfa, R_b, px, vx, yita = args
        cons = ({'type': 'ineq',
                 # 'fun': lambda x: m / ddt * (alfa * (R_b - px - vx * ddt) - vx) - m * (1 - yita) * vx - x[2] + x[3]})
                 'fun': lambda x: - vx - (1 - yita) * vx * ddt - ddt / m * (x[2] - x[3]) + alfa * (
                         R_b - px - vx * ddt)})
        return cons

    def con(self, args):
        m, ddt, alfa, px, py, vx, vy, p0x, p0y, r0, yita = args
        # cons = ({'type': 'ineq',
        #          'fun': lambda x: ((px - p0x) * vx + (py - p0y) * vy) / (((px - p0x) ** 2 + (py - p0y) ** 2) ** 0.5) + (1 - yita) * (vx + vy) * ddt + alfa * (((px - p0x) ** 2 + (py - p0y) ** 2) ** 0.5 + vx * ddt + vy * ddt - r0) + ddt / m * (x[2] - x[3] + x[0] - x[1])})
        cons = ({'type': 'ineq',
                 'fun': lambda x: (vx * (px + vx * ddt - p0x) + vy * (py + vy * ddt - p0y) + (
                         (1 - yita) * vx + 1 / m * (x[2] - x[3])) * ddt * (px + vx * ddt - p0x) + (
                                           (1 - yita) * vy + 1 / m * (x[0] - x[1])) * ddt * (
                                           py + vy * ddt - p0y)) / (((px + vx * ddt - p0x) ** 2 + (
                         py + vy * ddt - p0y) ** 2) ** 0.5) + alfa * (
                                          ((px + vx * ddt - p0x) ** 2 + (py + vy * ddt - p0y) ** 2) ** 0.5 - r0)})

        return cons

    def apply_coll_force(self, tmp_pos, action_n, p_velx, obst_pos, p_force):
        # 判断是否出界, 并施加拉力
        c = 0.6
        # for id, pos in enumerate(tmp_pos):
        #     tmp_pos[id] = self.agents[id].state.p_pos
        for id, pos in enumerate(tmp_pos):
            conss = []
            args = (action_n[id][1], action_n[id][2], action_n[id][3], action_n[id][4])  # action_n 上下右左的力
            abs_pos = np.abs(pos)
            # 0.8--0.85
            if (abs_pos > 0.85).any():
                X = pos[0]
                Y = pos[1]
                if X - 0.85 > 0:  # 從右邊出界
                    argsr = (1, 0.1, 1, 1, X, p_velx[id][0][0], 0.25)
                    conss.append(self.con_right(argsr))
                if X + 0.85 < 0:  # 從左邊出界
                    argsr = (1, 0.1, 1, -1, X, p_velx[id][0][0], 0.25)
                    conss.append(self.con_left(argsr))
                if Y - 0.85 > 0:  # 從上面出界
                    argsr = (1, 0.1, 1, 1, Y, p_velx[id][0][1], 0.25)
                    conss.append(self.con_up(argsr))
                if Y + 0.85 < 0:  # 從下面出界
                    argsr = (1, 0.1, 1, -1, Y, p_velx[id][0][1], 0.25)
                    conss.append(self.con_down(argsr))
            # 判断是否与障碍物、无人机发生碰撞，并施加拉力
            for b, agent_b in enumerate(self.agents):
                if b != id:
                    # da_tmp = np.linalg.norm(tmp_pos[id] - tmp_pos[b])
                    da_tmp = np.linalg.norm(tmp_pos[id] - tmp_pos[b])
                    if da_tmp < c + 0.1:
                        # if da_tmp < 0.3:
                        argsr = (
                            1, 0.1, 1, tmp_pos[id][0], tmp_pos[id][1], p_velx[id][0][0], p_velx[id][0][1],
                            tmp_pos[b][0],
                            tmp_pos[b][1], 0.1, 0.25)  # p0x=碰撞智能体x, p0y=碰撞智能体y, r0=0.05
                        conss.append(self.con(argsr))
            d_tmp = np.linalg.norm(tmp_pos[id] - np.array([0.3, 0.3]))
            if d_tmp < 0.4:
                # if d_tmp < 0.4:
                argsr = (1, 0.1, 1, tmp_pos[id][0], tmp_pos[id][1], p_velx[id][0][0], p_velx[id][0][1], 0.3, 0.3, 0.2,
                         0.25)  # p0x=0.4, p0y=0.4, r0=0.15
                conss.append(self.con(argsr))

            d1_tmp = np.linalg.norm(tmp_pos[id] - np.array([-0.6, 0.4]))
            # 0.88--0.5
            if d1_tmp < 0.35:
                # if d_tmp < 0.4:
                argsr = (1, 0.1, 1, tmp_pos[id][0], tmp_pos[id][1], p_velx[id][0][0], p_velx[id][0][1], -0.6, 0.4, 0.2,
                         0.25)  # p0x=0.4, p0y=0.4, r0=0.15
                conss.append(self.con(argsr))

            d2_tmp = np.linalg.norm(tmp_pos[id] - np.array([-0.7, 0.7]))
            # 0.74--0.3
            if d2_tmp < 0.3:
                # if d_tmp < 0.4:
                argsr = (1, 0.1, 1, tmp_pos[id][0], tmp_pos[id][1], p_velx[id][0][0], p_velx[id][0][1], -0.7, 0.7, 0.14,
                         0.25)  # p0x=0.4, p0y=0.4, r0=0.15
                conss.append(self.con(argsr))

            d3_tmp = np.linalg.norm(tmp_pos[id] - np.array([-0.7, 0.1]))
            # 0.74--0.3
            if d3_tmp < 0.3:
                # if d_tmp < 0.4:
                argsr = (1, 0.1, 1, tmp_pos[id][0], tmp_pos[id][1], p_velx[id][0][0], p_velx[id][0][1], -0.7, 0.1, 0.14,
                         0.25)  # p0x=0.4, p0y=0.4, r0=0.15
                conss.append(self.con(argsr))

            d4_tmp = np.linalg.norm(tmp_pos[id] - np.array([-0.3, 0.5]))
            # 0.74--0.3
            if d4_tmp < 0.3:
                # if d_tmp < 0.4:
                argsr = (1, 0.1, 1, tmp_pos[id][0], tmp_pos[id][1], p_velx[id][0][0], p_velx[id][0][1], -0.3, 0.5, 0.14,
                         0.25)  # p0x=0.4, p0y=0.4, r0=0.15
                conss.append(self.con(argsr))

            d5_tmp = np.linalg.norm(tmp_pos[id] - np.array([0.2, -0.4]))
            # 0.88--0.5
            if d5_tmp < 0.35:
                # if d_tmp < 0.4:
                argsr = (1, 0.1, 1, tmp_pos[id][0], tmp_pos[id][1], p_velx[id][0][0], p_velx[id][0][1], 0.2, -0.4, 0.2,
                         0.25)  # p0x=0.4, p0y=0.4, r0=0.15
                conss.append(self.con(argsr))

            d6_tmp = np.linalg.norm(tmp_pos[id] - np.array([0.5, -0.5]))
            # 0.74--0.3
            if d6_tmp < 0.3:
                # if d_tmp < 0.4:
                argsr = (1, 0.1, 1, tmp_pos[id][0], tmp_pos[id][1], p_velx[id][0][0], p_velx[id][0][1], 0.5, -0.5, 0.14,
                         0.25)  # p0x=0.4, p0y=0.4, r0=0.15
                conss.append(self.con(argsr))

            if conss:
                res = minimize(self.func(args), args, method='SLSQP', constraints=tuple(conss))
                p_force[id] = np.array([res.x[2] - res.x[3], res.x[0] - res.x[1]])
        return p_force

    #
    # def apply_coll_force(self, tmp_pos, obst_pos, p_force):
    #     # 判断是否与障碍物、无人机发生碰撞，并施加拉力
    #     """
    #     :param obst_pos:(list)障碍物数量x4的矩阵，【x_min,x_max,y_min,y_max】
    #     :return:
    #     """
    #     # 可优化(距离矩阵计算)
    #     c = 0.25
    #     for a, agent_a in enumerate(self.agents):
    #         for b, agent_b in enumerate(self.agents):
    #             self.dist_mat[a, b] = np.linalg.norm(tmp_pos[a] - tmp_pos[b])
    #             if self.dist_mat[a, b] < agent_a.size + agent_b.size:
    #                 d = np.linalg.norm(agent_a.state.p_pos - agent_b.state.p_pos)
    #                 aa = c * ((d - agent_a.r_cover) / (d - agent_a.size)) ** 2
    #                 p_force[a] += 0.6*aa * (agent_a.state.p_pos - agent_b.state.p_pos)
    #     for id, pos in enumerate(tmp_pos):
    #         for obst_id in range(len(obst_pos)):
    #             d_tmp = self.obst_dis(pos, obst_pos[obst_id])
    #             if d_tmp <= self.agents[id].size:
    #                 d = self.obst_dis(self.agents[id].state.p_pos, obst_pos[obst_id])
    #                 aa = c * ((d - self.agents[id].r_cover) / (d - self.agents[id].size)) ** 2
    #                 obst_x = 0.5 * (obst_pos[obst_id][1] - obst_pos[obst_id][0])
    #                 obst_y = 0.5 * (obst_pos[obst_id][3] - obst_pos[obst_id][2])
    #                 p_force[id] += 0.6*aa * (pos - np.array([obst_x, obst_y]))
    #
    #     return p_force

    def obst_dis(self, pos, obst_pos):
        x = pos[0]
        y = pos[1]
        xmin = obst_pos[0]
        xmax = obst_pos[1]
        ymin = obst_pos[2]
        ymax = obst_pos[3]
        if x <= xmin:
            if y >= ymax:
                aa = np.array([xmin, ymax])
                d = np.linalg.norm(pos - aa)
            elif ymin < y < ymax:
                d = np.abs(xmin - x)
            else:
                aa = np.array([xmin, ymin])
                d = np.linalg.norm(pos - aa)
        elif x >= xmax:
            if y >= ymax:
                aa = np.array([xmax, ymax])
                d = np.linalg.norm(pos - aa)
            elif ymin < y < ymax:
                d = np.abs(xmax - x)
            else:
                aa = np.array([xmax, ymin])
                d = np.linalg.norm(pos - aa)
        else:
            if y >= ymax:
                d = np.abs(y - ymax)
            elif ymin < y < ymax:
                d = 0
            else:
                d = np.abs(y - ymin)
        return d
