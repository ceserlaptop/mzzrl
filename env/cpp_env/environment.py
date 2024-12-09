# 1.改造step函数, 将reward和done脱离循环, 只执行一次, 并复制到所有agent
# 2.改造render函数, 在geometry list后增加2*num_agents个geom, 用于刻画uav的覆盖圆和通信圆
# 3.每轮render都重新导入landmark的颜色已显示当前的覆盖程度
# 4.改造动作空间维度, 将2*dim_p+1 改为 dim_p维度, u[0] = action[0], u[1] = action[1]
# 5.对通信模块进行删除

import gym
from gym import spaces
# from gym.envs.registration import EnvSpec
import numpy as np
from sklearn.cluster import KMeans
from alg.utils import get_square_vertices


# from multiagent_env.multiagent import rendering


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0
        # 判定当前环境是否重置
        self.isrest = False

        self.stepnum = 0  # the num of state convert

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space

            u_action_space = spaces.Discrete(1 + 2 * world.dim_p)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space

            # total action space
            self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def area_partition(self):
        pois_list = [poi.state.p_pos for poi in self.world.landmarks if not poi.done]
        if len(pois_list) < self.n:
            return
        kmeans = KMeans(self.n, random_state=0).fit(pois_list)
        labels = kmeans.labels_
        # 给每个poi分配类标签
        i = 0
        for poi in self.world.landmarks:
            if not poi.done:
                poi.label = labels[i]
                i += 1
        # 给智能体分配子区域
        centers = kmeans.cluster_centers_
        for ag in self.agents:
            dis_min = 100
            ag.target_id = -1
            for i, cen in enumerate(centers):
                dis = np.linalg.norm(ag.state.p_pos - cen)
                if dis <= dis_min:
                    dis_min = dis
                    ag.target_id = i
            centers[ag.target_id] = [10, 10]  # 将已经选择的区域中心置远，避免重复选择

    def step(self, action_input):
        # # todo 这里只用来调试，后期需要去掉
        # action_n = [np.array([13719068, 0.02047865, 0.7962253, 0.03812775, 0.00797753]),
        #            np.array([0.07508944, 0.7004558, 0.04798195, 0.01469667, 0.16177621]),
        #            np.array([0.0214441, 0.748268, 0.10931759, 0.01024642, 0.11072386]),
        #            np.array([0.03743383, 0.2150916, 0.6732575, 0.04681827, 0.02739876]), ]
        action_n = [arr[:5] for arr in action_input]
        action_h = [arr[5:] for arr in action_input]
        obs_n = []
        self.agents = self.world.policy_agents

        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])

        # advance world state
        self.world.step(action_n)

        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        reward_n = []
        for i in range(len(self.agents)):
            reward_agent = self._get_reward(self.agents[i], action_h[i])
            reward_n.append(reward_agent)
        print(reward_n)
        # reward_n = [self._get_reward(self.agents[0]) * self.n] * self.n

        done_n = [self._get_done(self.agents[0])] * self.n
        self.isrest = False

        return obs_n, reward_n, done_n

    # def step(self, action_n,train_ep):
    #     obs_n = []
    #     reward_n = []
    #     done_n = []
    #     self.agents = self.world.policy_agents
    #     # set action for each agent
    #     for i, agent in enumerate(self.agents):
    #         self._set_action(action_n[i], agent, self.action_space[i])
    #     # advance world state
    #     self.world.step(action_n)
    #     # record observation for each agent
    #     for agent in self.agents:
    #         obs_n.append(self._get_obs(agent))
    #         # reward_n = [self._get_reward(self.agents[0],train_ep) * self.n] * self.n
    #         # done_n = [self._get_done(self.agents[0])] * self.n
    #         reward_n.append(self._get_reward(agent, train_ep))
    #         done_n.append(self._get_done(agent))
    #     reward = np.sum(reward_n)
    #     if self.shared_reward:
    #         reward_n = [reward] * self.n
    #     self.isrest = False
    #
    #     return obs_n, reward_n, done_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        self.stepnum = 0
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []

        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        self.isrest = True
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    def _get_state(self, agent):
        if self.state_callback is None:
            return np.zeros(0)
        return self.state_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent, action_h):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world, action_h)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action

        if agent.movable:
            agent.action.u[0] = action[1] - action[2]
            agent.action.u[1] = action[3] - action[4]
            sensitivity = 2.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                # from gym.envs.classic_control import rendering
                from env.cpp_env import rendering
                self.viewers[i] = rendering.Viewer(700, 700)

        # create rendering geometry创造render的显存
        if self.render_geoms is None:
            from env.cpp_env import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                if 'field' not in entity.name:
                    geom = rendering.make_circle(entity.size)
                    xform = rendering.Transform()
                else:
                    entity.view_vertices = get_square_vertices(entity.state.p_pos, 0.2)  # 根据每一个目标点的位置创建一个正方形
                    geom = rendering.make_polygon(entity.view_vertices, filled=True)
                    xform = rendering.Transform()
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)
            #############################################################
            # 新增代码, 绘制覆盖范围
            for agent in self.world.agents:
                geom_cover = rendering.make_circle(agent.r_cover)
                xform = rendering.Transform()
                geom_cover.set_color(*agent.cover_color, alpha=0.5)
                geom_cover.add_attr(xform)
                self.render_geoms.append(geom_cover)
                self.render_geoms_xform.append(xform)
            # 新增代码, 绘制uav之间的通信线, 若两个通信则画个线, 一共需要
            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)
        ##################################################################
        # 新增代码, 每轮显示agent的r_cover, r_comm, poi的是否完成
        for geom, entity in zip(self.render_geoms, self.world.entities):
            geom.set_color(*entity.color)
        results = []
        for i in range(len(self.viewers)):
            # update bounds to center around agent
            cam_range = 1.5
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range, pos[1] + cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                if 'field' not in entity.name:
                    self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            ###########################################################
            for e, agent in enumerate(self.agents):  # 绘制覆盖范围
                self.render_geoms_xform[e + len(self.world.entities)].set_translation(*agent.state.p_pos)

            self.viewers[i].draw_line([-1, -1], [1, -1])
            self.viewers[i].draw_line([-1, 1], [1, 1])
            self.viewers[i].draw_line([-1, -1], [-1, 1])
            self.viewers[i].draw_line([1, 1], [1, -1])
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array=(mode == 'rgb_array')))

        return results

    def random_actions(self):
        action_n = [np.random.rand(5) for _ in range(4)]
        return action_n

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i + env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
