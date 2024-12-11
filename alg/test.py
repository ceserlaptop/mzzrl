import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

# --- 超参数 ---
GAMMA = 0.99
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 2000
BATCH_SIZE = 64
TARGET_UPDATE = 10
LEARNING_RATE = 0.001
MEMORY_SIZE = 10000
NUM_EPISODES = 5000

# --- 环境参数 ---
LARGE_CIRCLE_RADIUS = 1500
SMALL_CIRCLE_RADIUS = 800
UAV_INITIAL_POS = np.array([12000, 5000])
UAV_INITIAL_HEADING = np.pi  # 朝向x轴负方向
UAV_SPEED = 300
RADAR_RANGE = 1600
RADAR_ANGLE = 30 * np.pi / 180
TARGET_CIRCLE_RADIUS = 800  # Target可能出现的区域的半径

# --- 动作空间 ---
ACTION_STRAIGHT = 0
ACTION_TURN_LEFT = 1
ACTION_TURN_RIGHT = 2
NUM_ACTIONS = 3
TURN_ANGLE = 5 * np.pi / 180  # 每次转弯的角度


# --- DQN 网络 ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# --- 经验回放 ---
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# --- 环境 ---
class Environment:
    def __init__(self, uav_num):
        self.uav_num = uav_num
        self.last_distance = [float('inf')] * uav_num
        self.uav_positions = np.array([UAV_INITIAL_POS] * uav_num)
        self.uav_headings = np.array([UAV_INITIAL_HEADING] * uav_num)
        self.target_pos = np.array(
            [0, 0])  # Placeholder, will be updated later, but represents the center of the small circle
        self.steps = 0
        self.done = False
        self.large_circle_reached = [False] * uav_num
        self.large_circle_reach_time = [float('inf')] * uav_num

    def reset(self):
        # Reset UAV positions and headings
        self.uav_positions = np.array([UAV_INITIAL_POS] * self.uav_num)
        self.uav_headings = np.array([UAV_INITIAL_HEADING] * self.uav_num)

        # Randomly place the small circle's center within the reachable area
        # Considering the UAVs start at [12000, 5000] and need to reach the large circle (radius 1500)
        # The small circle's center should be within a reasonable range
        max_dist_from_start = 12000  # This is just an example and might need adjustment

        while True:
            # Generate a random position for the center of the small circle
            # The center can be anywhere within a circle of radius max_dist_from_start - LARGE_CIRCLE_RADIUS
            # centered at the origin, as a simplification
            angle = random.uniform(0, 2 * np.pi)
            # radius = random.uniform(0, max_dist_from_start - LARGE_CIRCLE_RADIUS)
            # self.target_pos = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            self.target_pos = np.array([0, 0])

            # Check if both UAVs can reach the large circle from their starting positions
            dist_uav1_to_large_circle = np.linalg.norm(UAV_INITIAL_POS - self.target_pos)
            dist_uav2_to_large_circle = np.linalg.norm(
                UAV_INITIAL_POS - self.target_pos)  # Assuming the second UAV also starts at the same position

            # Ensure the large circle is reachable by both UAVs
            if dist_uav1_to_large_circle >= LARGE_CIRCLE_RADIUS and dist_uav2_to_large_circle >= LARGE_CIRCLE_RADIUS:
                break

        self.steps = 0
        self.done = False
        self.large_circle_reached = [False] * self.uav_num
        self.large_circle_reach_time = [float('inf')] * self.uav_num
        return self.get_state()

    def step(self, actions):
        self.steps += 1
        for i in range(self.uav_num):
            if actions[i] == ACTION_TURN_LEFT:
                self.uav_headings[i] = (self.uav_headings[i] + TURN_ANGLE) % (2 * np.pi)
            elif actions[i] == ACTION_TURN_RIGHT:
                self.uav_headings[i] = (self.uav_headings[i] - TURN_ANGLE) % (2 * np.pi)

            self.uav_positions[i][0] += UAV_SPEED * np.cos(self.uav_headings[i])
            self.uav_positions[i][1] += UAV_SPEED * np.sin(self.uav_headings[i])

            # Check if reached large circle
            if not self.large_circle_reached[i] and np.linalg.norm(
                    self.uav_positions[i] - self.target_pos) <= LARGE_CIRCLE_RADIUS:
                self.large_circle_reached[i] = True
                self.large_circle_reach_time[i] = self.steps

        # Check if done (both UAVs reached large circle or max steps reached)
        if all(self.large_circle_reached) or self.steps >= 1000:
            self.done = True

        reward = self.calculate_reward()
        next_state = self.get_state()
        return next_state, reward, self.done

    def get_state(self):
        state = []
        for i in range(self.uav_num):
            # Use the center of the small circle as the reference point for the state
            state.append(self.uav_positions[i][0] - self.target_pos[0])
            state.append(self.uav_positions[i][1] - self.target_pos[1])
            state.append(self.uav_headings[i])

            # Add relative position of the other UAV
            if i == 0:
                state.append(self.uav_positions[1][0] - self.uav_positions[0][0])
                state.append(self.uav_positions[1][1] - self.uav_positions[0][1])
            else:
                state.append(self.uav_positions[0][0] - self.uav_positions[1][0])
                state.append(self.uav_positions[0][1] - self.uav_positions[1][1])

        return np.array(state)

    def calculate_reward(self):
        reward = 0

        # Reward 1: Time difference to reach the large circle
        if all(self.large_circle_reached):
            time_diff_penalty = self.large_circle_reach_time[0] - self.large_circle_reach_time[1]
            reward += 40-time_diff_penalty * 0.1
        else:
            reward -= 0.005
        # time_diff_penalty = abs(self.large_circle_reach_time[0] - self.large_circle_reach_time[1])
        # reward -= time_diff_penalty * 0.01

        for i in range(self.uav_num):
            dist_to_target = np.linalg.norm(self.uav_positions[i] - self.target_pos)
            reward += (self.last_distance[i]-dist_to_target) * 0.0005  # Adjust the scaling factor as needed
            self.last_distance[i] = dist_to_target

        if self.done:
            # Reward 2: Overlap area of radar coverage within the small circle (only calculated when both reach)
            if all(self.large_circle_reached):
                overlap_area_reward = self.calculate_overlap_area()
                reward += overlap_area_reward * 0.001  # Scale the reward

            # Reward 3: Favorable heading for target detection (only calculated when both reach)
            if all(self.large_circle_reached):
                heading_reward = self.calculate_heading_reward()
                reward += heading_reward

        return reward

    def calculate_overlap_area(self):
        # Approximate the overlap area using a grid-based method
        grid_size = 50
        count = 0
        total = 0
        for x in np.linspace(self.target_pos[0] - SMALL_CIRCLE_RADIUS, self.target_pos[0] + SMALL_CIRCLE_RADIUS,
                             grid_size):
            for y in np.linspace(self.target_pos[1] - SMALL_CIRCLE_RADIUS, self.target_pos[1] + SMALL_CIRCLE_RADIUS,
                                 grid_size):
                point = np.array([x, y])
                if np.linalg.norm(point - self.target_pos) <= SMALL_CIRCLE_RADIUS:
                    total += 1
                    if self.is_in_radar(self.uav_positions[0], self.uav_headings[0], point) and \
                            self.is_in_radar(self.uav_positions[1], self.uav_headings[1], point):
                        count += 1

        if total == 0:
            return 0
        else:
            return count / total * (np.pi * SMALL_CIRCLE_RADIUS ** 2)  # Scale by the area of the small circle

    def is_in_radar(self, uav_pos, uav_heading, point):
        # Check if the point is within the radar range
        if np.linalg.norm(uav_pos - point) > RADAR_RANGE:
            return False

        # Check if the point is within the radar sector
        relative_pos = point - uav_pos
        angle = np.arctan2(relative_pos[1], relative_pos[0])
        angle_diff = abs(angle - uav_heading)
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)

        return angle_diff <= RADAR_ANGLE / 2

    def calculate_heading_reward(self):
        reward = 0
        for i in range(self.uav_num):
            # Project the UAV's position to the large circle
            projected_uav_pos, projected_uav_heading = self.project_to_large_circle(self.uav_positions[i],
                                                                                    self.uav_headings[i],
                                                                                    self.target_pos)

            # Find intersection points of the projected velocity vector with the small circle
            intersections = self.find_intersections(projected_uav_pos, projected_uav_heading, self.target_pos,
                                                    SMALL_CIRCLE_RADIUS)

            if len(intersections) > 0:
                # Consider the first intersection point
                intersection_point = intersections[0]

                # Calculate the angle of the intersection point relative to the center of the small circle
                angle_to_center = np.arctan2(intersection_point[1] - self.target_pos[1],
                                             intersection_point[0] - self.target_pos[0])

                # Normalize the angle to be between 0 and 2*pi
                angle_to_center = (angle_to_center + 2 * np.pi) % (2 * np.pi)

                # Define the favorable angle ranges (45-135 degrees and 225-315 degrees)
                favorable_range1 = (np.pi / 4, 3 * np.pi / 4)  # 45-135 degrees
                favorable_range2 = (5 * np.pi / 4, 7 * np.pi / 4)  # 225-315 degrees

                # Check if the intersection angle is within the favorable ranges
                if favorable_range1[0] <= angle_to_center <= favorable_range1[1] or \
                        favorable_range2[0] <= angle_to_center <= favorable_range2[1]:
                    reward += 5  # Higher reward for being in the favorable range
                else:
                    reward += 1  # Lower reward for being outside the favorable range
            else:
                reward -= 2  # Penalty for no intersection

        return reward

    def project_to_large_circle(self, uav_pos, uav_heading, target_pos):
        """
        Projects the UAV's position to the point where it would reach the large circle,
        assuming it continues with its current heading.
        """
        # Calculate the direction vector of the UAV's velocity
        direction_vector = np.array([np.cos(uav_heading), np.sin(uav_heading)])

        # Vector from the target position (center of the large circle) to the UAV's position
        v = uav_pos - target_pos

        # Coefficients for the quadratic equation
        a = np.dot(direction_vector, direction_vector)
        b = 2 * np.dot(v, direction_vector)
        c = np.dot(v, v) - LARGE_CIRCLE_RADIUS ** 2

        # Calculate the discriminant
        discriminant = b ** 2 - 4 * a * c

        if discriminant >= 0:
            # Calculate the positive solution for t (distance to travel)
            t = (-b + np.sqrt(discriminant)) / (2 * a)

            # Project the UAV's position to the large circle
            projected_uav_pos = uav_pos + t * direction_vector

            return projected_uav_pos, uav_heading
        else:
            # UAV is not moving towards the large circle (should not happen in normal cases)
            return uav_pos, uav_heading

    def find_intersections(self, uav_pos, uav_heading, circle_center, circle_radius):
        # Calculate the direction vector of the UAV's velocity
        direction_vector = np.array([np.cos(uav_heading), np.sin(uav_heading)])

        # Vector from the circle center to the UAV's position
        v = uav_pos - circle_center

        # Coefficients for the quadratic equation
        a = np.dot(direction_vector, direction_vector)
        b = 2 * np.dot(v, direction_vector)
        c = np.dot(v, v) - circle_radius ** 2

        # Calculate the discriminant
        discriminant = b ** 2 - 4 * a * c

        intersections = []
        if discriminant >= 0:
            # Calculate the two solutions for t
            t1 = (-b - np.sqrt(discriminant)) / (2 * a)
            t2 = (-b + np.sqrt(discriminant)) / (2 * a)

            if t1 > 0:  # Consider only future intersections
                intersection1 = uav_pos + t1 * direction_vector
                intersections.append(intersection1)
            if t2 > 0 and t2 != t1:
                intersection2 = uav_pos + t2 * direction_vector
                intersections.append(intersection2)

        return intersections


# --- DQN Agent ---
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0

    def select_action(self, state, available_actions=[0, 1, 2]):
        sample = random.random()
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                        math.exp(-1. * self.steps_done / EPSILON_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                q_values = self.policy_net(torch.from_numpy(state).float())
                # Select from available actions only
                valid_q_values = [q_values[i].item() if i in available_actions else float('-inf') for i in
                                  range(len(q_values))]
                return np.argmax(valid_q_values)
        else:
            return random.choice(available_actions)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

        batch_state = torch.tensor(batch_state, dtype=torch.float)
        batch_action = torch.tensor(batch_action, dtype=torch.long).unsqueeze(1)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float)
        batch_next_state = torch.tensor(batch_next_state, dtype=torch.float)
        batch_done = torch.tensor(batch_done, dtype=torch.float)

        current_q_values = self.policy_net(batch_state).gather(1, batch_action)

        with torch.no_grad():
            max_next_q_values = self.target_net(batch_next_state).max(1)[0]
            expected_q_values = batch_reward + (1 - batch_done) * GAMMA * max_next_q_values

        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# --- 训练 ---
def train():
    env = Environment(uav_num=2)
    state_dim = env.get_state().shape[0]
    agent1 = DQNAgent(state_dim, NUM_ACTIONS)
    agent2 = DQNAgent(state_dim, NUM_ACTIONS)

    episode_rewards = []
    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action1 = agent1.select_action(state)
            action2 = agent2.select_action(state)

            next_state, reward, done = env.step([action1, action2])

            agent1.memory.push(state, action1, reward, next_state, done)
            agent2.memory.push(state, action2, reward, next_state, done)

            state = next_state
            total_reward += reward
            agent1.optimize_model()
            agent2.optimize_model()

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{NUM_EPISODES}, Reward: {total_reward:.2f}, Target: {env.target_pos}")

        if episode % TARGET_UPDATE == 0:
            agent1.target_net.load_state_dict(agent1.policy_net.state_dict())
            agent2.target_net.load_state_dict(agent2.policy_net.state_dict())

    print("Training finished.")
    return agent1, agent2, episode_rewards


# --- 绘制轨迹 ---
def plot_trajectory(env, agent1, agent2):
    state = env.reset()
    done = False
    trajectory1 = [env.uav_positions[0].copy()]
    trajectory2 = [env.uav_positions[1].copy()]
    radar_coverages = []

    while not done:
        action1 = agent1.select_action(state)
        action2 = agent2.select_action(state)
        next_state, _, done = env.step([action1, action2])
        state = next_state
        trajectory1.append(env.uav_positions[0].copy())
        trajectory2.append(env.uav_positions[1].copy())
        radar_coverages.append((env.uav_positions[0].copy(), env.uav_headings[0], env.uav_positions[1].copy(), env.uav_headings[1]))

    trajectory1 = np.array(trajectory1)
    trajectory2 = np.array(trajectory2)

    # --- 创建两个子图 ---
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # --- 第一个子图：全局轨迹图 ---
    ax = axs[0]
    # 绘制轨迹
    ax.plot(trajectory1[:, 0], trajectory1[:, 1], 'r-', label='UAV 1')
    ax.plot(trajectory2[:, 0], trajectory2[:, 1], 'b-', label='UAV 2')

    # 绘制初始位置
    ax.plot(UAV_INITIAL_POS[0], UAV_INITIAL_POS[1], 'ro')
    ax.plot(UAV_INITIAL_POS[0], UAV_INITIAL_POS[1], 'bo')

    # 绘制大圆、小圆
    large_circle = Circle(env.target_pos, LARGE_CIRCLE_RADIUS, color='blue', fill=False)
    ax.add_patch(large_circle)
    small_circle = Circle(env.target_pos, SMALL_CIRCLE_RADIUS, color='green', fill=False)
    ax.add_patch(small_circle)

    # 绘制隐藏目标点
    ax.plot(env.target_pos[0], env.target_pos[1], 'go', markersize=1, label='Hidden Target')

    # 设置坐标轴范围
    ax.set_xlim(-20000, 20000)
    ax.set_ylim(-20000, 20000)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'UAV Trajectories (Target: {env.target_pos})')
    ax.legend()
    ax.set_aspect('equal', 'box')
    ax.grid(True)

    # --- 第二个子图：大圆处放大图 ---
    ax = axs[1]
    # 绘制大圆、小圆
    large_circle = Circle(env.target_pos, LARGE_CIRCLE_RADIUS, color='blue', fill=False)
    ax.add_patch(large_circle)
    small_circle = Circle(env.target_pos, SMALL_CIRCLE_RADIUS, color='green', fill=False)
    ax.add_patch(small_circle)

    # 绘制隐藏目标点
    ax.plot(env.target_pos[0], env.target_pos[1], 'go', markersize=1, label='Hidden Target')

    # 绘制最后一步到达大圆的位置及朝向
    uav1_pos, uav1_heading, uav2_pos, uav2_heading = radar_coverages[-1] # 取最后一步
    projected_uav1_pos, projected_uav1_heading = env.project_to_large_circle(uav1_pos, uav1_heading, env.target_pos)
    projected_uav2_pos, projected_uav2_heading = env.project_to_large_circle(uav2_pos, uav2_heading, env.target_pos)

    ax.plot(projected_uav1_pos[0], projected_uav1_pos[1], 'rx')
    ax.plot(projected_uav2_pos[0], projected_uav2_pos[1], 'bx')

    # 绘制朝向箭头
    arrow_length = 200
    ax.arrow(projected_uav1_pos[0], projected_uav1_pos[1], arrow_length * np.cos(projected_uav1_heading), arrow_length * np.sin(projected_uav1_heading), color='red', head_width=100, head_length=100)
    ax.arrow(projected_uav2_pos[0], projected_uav2_pos[1], arrow_length * np.cos(projected_uav2_heading), arrow_length * np.sin(projected_uav2_heading), color='blue', head_width=100, head_length=100)

    # 绘制雷达覆盖区域（最后一步）
    radar_polygon1 = get_radar_polygon(uav1_pos, uav1_heading)
    ax.add_patch(Polygon(radar_polygon1, closed=True, color='red', alpha=0.3))
    radar_polygon2 = get_radar_polygon(uav2_pos, uav2_heading)
    ax.add_patch(Polygon(radar_polygon2, closed=True, color='blue', alpha=0.3))

    # 设置坐标轴范围
    margin = 500
    min_x = min(projected_uav1_pos[0], projected_uav2_pos[0], env.target_pos[0] - LARGE_CIRCLE_RADIUS) - margin
    max_x = max(projected_uav1_pos[0], projected_uav2_pos[0], env.target_pos[0] + LARGE_CIRCLE_RADIUS) + margin
    min_y = min(projected_uav1_pos[1], projected_uav2_pos[1], env.target_pos[1] - LARGE_CIRCLE_RADIUS) - margin
    max_y = max(projected_uav1_pos[1], projected_uav2_pos[1], env.target_pos[1] + LARGE_CIRCLE_RADIUS) + margin

    ax.set_xlim(-1800, 1800)
    ax.set_ylim(-1800, 1800)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'UAV at Large Circle (Target: {env.target_pos})')
    ax.legend()
    ax.set_aspect('equal', 'box')
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def get_radar_polygon(uav_pos, uav_heading):
    p1 = uav_pos
    p2 = uav_pos + RADAR_RANGE * np.array(
        [np.cos(uav_heading - RADAR_ANGLE / 2), np.sin(uav_heading - RADAR_ANGLE / 2)])
    p3 = uav_pos + RADAR_RANGE * np.array(
        [np.cos(uav_heading + RADAR_ANGLE / 2), np.sin(uav_heading + RADAR_ANGLE / 2)])
    return np.array([p1, p2, p3])


# --- 主程序 ---
if __name__ == "__main__":
    trained_agent1, trained_agent2, episode_rewards = train()

    # Save the trained models (optional)
    torch.save(trained_agent1.policy_net.state_dict(), 'agent1_policy_net.pth')
    torch.save(trained_agent2.policy_net.state_dict(), 'agent2_policy_net.pth')

    # You can further analyze the episode_rewards, e.g., plot them to see the learning progress.
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.show()

    # Load the trained models for testing
    state_dim = 10  # 2 UAVs * (x, y, heading)
    loaded_agent1 = DQNAgent(state_dim, NUM_ACTIONS)
    loaded_agent2 = DQNAgent(state_dim, NUM_ACTIONS)
    loaded_agent1.policy_net.load_state_dict(torch.load('agent1_policy_net.pth'))
    loaded_agent2.policy_net.load_state_dict(torch.load('agent2_policy_net.pth'))
    loaded_agent1.policy_net.eval()  # Set to evaluation mode
    loaded_agent2.policy_net.eval()

    # Test with a random target position
    test_env = Environment(uav_num=2)
    plot_trajectory(test_env, loaded_agent1, loaded_agent2)