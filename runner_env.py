import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class RunnerEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Actions: left, right, jump, duck, nothing
        self.action_space = spaces.Discrete(5)

        # State: player lane, obstacle distance
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(2,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None):
        self.player_lane = 1   # 0,1,2
        self.obstacle_lane = random.randint(0, 2)
        self.obstacle_dist = 10
        self.done = False
        return np.array([self.player_lane, self.obstacle_dist]), {}

    def step(self, action):
        reward = 0

        # Move player
        if action == 0 and self.player_lane > 0:
            self.player_lane -= 1
        elif action == 1 and self.player_lane < 2:
            self.player_lane += 1

        # Obstacle moves closer
        self.obstacle_dist -= 1

        # Collision
        if self.obstacle_dist == 0:
            if self.player_lane == self.obstacle_lane:
                reward = -100
                self.done = True
            else:
                reward = 10
                self.obstacle_dist = 10
                self.obstacle_lane = random.randint(0, 2)

        reward += 1  # survival reward

        state = np.array([self.player_lane, self.obstacle_dist])
        return state, reward, self.done, False, {}

    def render(self):
        print(f"Lane: {self.player_lane}, Obstacle: {self.obstacle_lane}, Dist: {self.obstacle_dist}")
