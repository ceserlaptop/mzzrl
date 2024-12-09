import math
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd
import numpy as np
import time
import os


# import env.cpp_env.scenarios as scenarios
# from env.cpp_env.environment import MultiAgentEnv


def normalize_angle(angle):
    """Normalize an angle in degree to :math:`[0, 360)`."""
    return (angle + 180) % 360.0 - 180.0


# environment
def angle_diff(angle1, angle2):
    """Calculate the difference between two angles."""
    return abs(normalize_angle(angle1 - angle2))


# coverage_0
def angle_sub(angle1, angle2):
    return ((angle1 - angle2) + 180.0) % 360.0 - 180.0


def eva_cost(R_pos, head_angle, T_pos):
    head_angle = np.radians(head_angle)
    # 计算RR_向量与向量RT之间的夹角
    RR_ = np.array([math.cos(head_angle), math.sin(head_angle)])
    RT = T_pos - R_pos
    d_RT = np.linalg.norm(RT)
    cos_ = np.dot(RR_, RT) / d_RT
    theta = np.arccos(cos_)
    s = (18 * theta / math.pi - 6)
    angle_cost = s / (1 + abs(s)) + 1
    return angle_cost / 4 + d_RT


def polar2cartesian(rho, phi):
    r"""Convert polar coordinates to cartesian coordinates **in degrees**, element-wise.

    .. math::
        \operatorname{polar2cartesian} ( \rho, \phi ) = \left( \rho \cos_{\text{deg}} ( \phi ), \rho \sin_{\text{deg}} ( \phi ) \right)
    """  # pylint: disable=line-too-long

    phi_rad = np.deg2rad(phi)
    return rho * np.array([np.cos(phi_rad), np.sin(phi_rad)])


def arctan2_deg(y, x):
    r"""Element-wise arc tangent of y/x **in degrees**.

    .. math::
        \operatorname{arctan2}_{\text{deg}} ( y, x ) = \frac{180}{\pi} \arctan \left( \frac{y}{x} \right)
    """

    return np.rad2deg(np.arctan2(y, x))


def arcsin_deg(x):
    r"""Trigonometric inverse sine **in degrees**, element-wise.

    .. math::
        \arcsin_{\text{deg}} ( x ) = \frac{180}{\pi} \arcsin ( x )
    """

    return np.rad2deg(np.arcsin(x))


# CoverageWorld
def point_to_line_distance(A, B, C):
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C

    # 计算AB向量
    AB = B - A
    # 计算AC向量
    AC = C - A

    # 计算AC在AB上的投影点D
    projection_length = np.dot(AC, AB) / np.dot(AB, AB)
    D = A + projection_length * AB

    # 计算C到D的距离
    distance = np.linalg.norm(C - D)

    # 判断投影点是否在AB线段上
    in_plag = min(x1, x2) <= x3 <= max(x1, x2) and min(y1, y2) <= y3 <= max(y1, y2)

    return distance, in_plag


def get_square_vertices(center_point, size):
    x, y = center_point
    half_size = size / 2
    vertices = np.array([
        [x - half_size, y - half_size],
        [x - half_size, y + half_size],
        [x + half_size, y + half_size],
        [x + half_size, y - half_size],
        [x - half_size, y - half_size]
    ])
    return vertices


def get_rgb(file_path):
    data = pd.read_csv(file_path)
    data_values = data.values
    # 绘制伪彩图
    cmap = plt.cm.viridis
    norm = Normalize(vmin=np.min(data_values), vmax=np.max(data_values))

    # 创建伪彩图
    # plt.figure(figsize=(6, 5))
    # plt.imshow(data, cmap=cmap, norm=norm)
    # plt.colorbar(label="Value")
    # plt.title("Map")
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.show()

    # 提取 RGB 数据
    rgb_data = cmap(norm(data))[:, :, :3]  # 获取 RGB 值 (去掉 alpha 通道)
    # rgb_data = (rgb_data * 255).astype(np.uint8)  # 转为 0-255 整数
    # 保存 RGB 数据为文件
    # rgb_df = pd.DataFrame(rgb_data.reshape(-1, 3), columns=["R", "G", "B"])
    # rgb_df.to_csv("rgb_data.csv", index=False)
    # print("RGB 数据已导出为 rgb_data.csv")

    return rgb_data.reshape(-1, 3)


def choose_point(agent, world_field_strength):
    min_distence = [[[None, np.inf], [None, np.inf]], [[None, np.inf], [None, np.inf]], [[None, np.inf], [None, np.inf]], [[None, np.inf], [None, np.inf]]]
    for field in world_field_strength:
        distence = calculate_distance(field.state.p_pos, agent.state.p_pos)
        angel = calculate_angle(field.state.p_pos, agent.state.p_pos)
        index = (angel-(np.pi/4))/(np.pi/2)
        if index < 0:
            if distence < min_distence[3][0][1]:
                min_distence[3][0][0] = field
                min_distence[3][0][1] = distence
            elif distence < min_distence[3][1][1]:
                min_distence[3][1][0] = field
                min_distence[3][1][1] = distence
            else:
                pass
        else:
            index = int(index)
            if distence < min_distence[index][0][1]:
                min_distence[index][0][0] = field
                min_distence[index][0][1] = distence
            elif distence < min_distence[index][1][1]:
                min_distence[index][1][0] = field
                min_distence[index][1][1] = distence
            else:
                pass
    return min_distence


def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


# 从point1指向point2的弧度制角度
def calculate_angle(point1, point2):
    # 计算 y 和 x 的差值
    delta_y = point2[1] - point1[1]
    delta_x = point2[0] - point1[0]
    # 使用 atan2 计算角度（弧度）
    angle_radians = math.atan2(delta_y, delta_x)
    # 确保弧度为非负值（0 到 2π）
    angle_radians = angle_radians % (2 * math.pi)
    return angle_radians


def calculate_pos(env_size, scope):
    border_distance = scope/env_size
    x = np.arange(-scope+border_distance, scope, 2*border_distance)
    y = np.arange(-scope+border_distance, scope, 2*border_distance)
    X, Y = np.meshgrid(x, y)
    x = X.reshape(env_size*env_size, )
    y = Y.reshape(env_size*env_size, )
    pos_pois = np.array([x, y])
    return pos_pois


def create_dir(scenario_name):
    """构造目录 ./scenario_name/experiment_name/(plots, policy, buffer)"""
    scenario_path = "./" + scenario_name
    if not os.path.exists(scenario_path):
        os.mkdir(scenario_path)

    tm_struct = time.localtime(time.time())
    experiment_name = scenario_name + "_%02d_%02d_%02d_%02d" % (tm_struct[1], tm_struct[2], tm_struct[3], tm_struct[4])
    experiment_path = os.path.join(scenario_path, experiment_name)
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    save_paths = [experiment_path + "/policy/", experiment_path + "/plots/"]
    for save_path in save_paths:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
    return save_paths[0], save_paths[1]


class Vector2D:  # pylint: disable=missing-function-docstring
    """2D Vector."""

    def __init__(self, vector=None, norm=None, angle=None, origin=None):
        self.origin = origin
        self._vector = None
        self._angle = None
        self._norm = None
        if vector is not None and norm is None and angle is None:
            self.vector = np.asarray(vector, dtype=np.float64)
        elif vector is None and norm is not None and angle is not None:
            self.angle = angle
            self.norm = norm
        else:
            raise ValueError

    @property
    def vector(self):
        if self._vector is None:
            self._vector = polar2cartesian(self._norm, self._angle)
        return self._vector

    @vector.setter
    def vector(self, value):
        self._vector = np.asarray(value, dtype=np.float64)
        self._norm = None
        self._angle = None

    @property
    def x(self):
        return self.vector[0]

    @property
    def y(self):
        return self.vector[-1]

    @property
    def endpoint(self):
        return self.origin + self.vector

    @endpoint.setter
    def endpoint(self, value):
        endpoint = np.asarray(value, dtype=np.float64)
        self.vector = endpoint - self.origin

    @property
    def angle(self):
        if self._angle is None:
            self._angle = arctan2_deg(self._vector[-1], self._vector[0])
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = normalize_angle(float(value))
        self._vector = None

    @property
    def norm(self):
        if self._norm is None:
            self._norm = np.linalg.norm(self._vector)
        return self._norm

    @norm.setter
    def norm(self, value):
        angle = self.angle
        self._norm = abs(float(value))
        self._vector = None
        if value < 0.0:
            self.angle = angle + 180.0

    def copy(self):
        return Vector2D(vector=self.vector.copy(), origin=self.origin)

    def __eq__(self, other):
        assert isinstance(other, Vector2D)

        return self.angle == other.angle

    def __ne__(self, other):
        return not self == other

    def __imul__(self, other):
        self.norm = self.norm * other

    def __add__(self, other):
        assert isinstance(other, Vector2D)

        return Vector2D(vector=self.vector + other.vector, origin=self.origin)

    def __sub__(self, other):
        assert isinstance(other, Vector2D)

        return Vector2D(vector=self.vector - other.vector, origin=self.origin)

    def __mul__(self, other):
        return Vector2D(norm=self.norm * other, angle=self.angle, origin=self.origin)

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        return Vector2D(norm=self.norm / other, angle=self.angle, origin=self.origin)

    def __pos__(self):
        return self

    def __neg__(self):
        return Vector2D(vector=-self.vector, origin=self.origin)

    def __array__(self):
        return self.vector.copy()
