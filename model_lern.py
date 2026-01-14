import pybullet as p
import pybullet_data
import numpy as np
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# ================== SETTINGS ==================
TRAIN_EPISODES = 150        # watch learning
TEST_EPISODES = 3           # watch final behavior
MAX_STEPS = 1500
RENDER = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================== TRACK ==================
TRACK_RADIUS_X = 25
TRACK_RADIUS_Y = 15
TRACK_WIDTH = 3.5
WALL_THICKNESS = 0.15
WALL_HEIGHT = 0.6
SEGMENTS = 500

def build_track():
    for i in range(SEGMENTS):
        t1 = 2 * math.pi * i / SEGMENTS
        t2 = 2 * math.pi * (i + 1) / SEGMENTS

        x1 = TRACK_RADIUS_X * math.cos(t1)
        y1 = TRACK_RADIUS_Y * math.sin(t1)
        x2 = TRACK_RADIUS_X * math.cos(t2)
        y2 = TRACK_RADIUS_Y * math.sin(t2)

        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        yaw = math.atan2(dy, dx)

        nx = -dy / length
        ny = dx / length

        for side in [1, -1]:
            cx = (x1 + x2) / 2 + side * nx * TRACK_WIDTH
            cy = (y1 + y2) / 2 + side * ny * TRACK_WIDTH

            col = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[length/2, WALL_THICKNESS, WALL_HEIGHT]
            )
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[length/2, WALL_THICKNESS, WALL_HEIGHT],
                rgbaColor=[0.3, 0.3, 0.3, 1]
            )

            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[cx, cy, WALL_HEIGHT],
                baseOrientation=p.getQuaternionFromEuler([0, 0, yaw])
            )

# ================== ENV ==================
class CarPlusEnv:
    def __init__(self):
        p.connect(p.GUI if RENDER else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        p.loadURDF("plane.urdf")
        build_track()

        p.resetDebugVisualizerCamera(
            cameraDistance=6,
            cameraYaw=60,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0]
        )

        self.car = None
        self.reset()

    def reset(self):
        if self.car is not None:
            p.removeBody(self.car)

        self.car = p.loadURDF("racecar/racecar.urdf", [0, 0, 1.0])
        self.steps = 0

        pos, _ = p.getBasePositionAndOrientation(self.car)
        return np.array([pos[0], pos[1]], dtype=np.float32)

    def step(self, steering, throttle):
        # steering (front wheels)
        p.setJointMotorControl2(self.car, 0, p.POSITION_CONTROL, targetPosition=steering)
        p.setJointMotorControl2(self.car, 2, p.POSITION_CONTROL, targetPosition=steering)

        # throttle (rear wheels)
        p.setJointMotorControl2(self.car, 8, p.VELOCITY_CONTROL, targetVelocity=throttle)
        p.setJointMotorControl2(self.car, 10, p.VELOCITY_CONTROL, targetVelocity=throttle)

        p.stepSimulation()
        if RENDER:
            time.sleep(1 / 60)

        pos, _ = p.getBasePositionAndOrientation(self.car)
        self.steps += 1

        # reward: stay near center
        reward = 1.0 - abs(pos[1]) * 0.1

        done = abs(pos[1]) > 7 or self.steps > MAX_STEPS
        if done:
            reward -= 20

        return np.array([pos[0], pos[1]], dtype=np.float32), reward, done

# ================== POLICY ==================
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.log_std = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        mean = self.net(x)
        std = self.log_std.exp()
        return mean, std

# ================== MAIN ==================
env = CarPlusEnv()
policy = Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=3e-4)

print("\n=== TRAINING (WATCH THE CAR LEARN) ===\n")

for episode in range(TRAIN_EPISODES):
    state = torch.tensor(env.reset(), device=device)
    episode_reward = 0

    for _ in range(MAX_STEPS):
        mean, std = policy(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        steering = torch.tanh(action[0]).item()
        throttle = 20 + 15 * torch.sigmoid(action[1]).item()

        next_state_np, reward, done = env.step(steering, throttle)
        next_state = torch.tensor(next_state_np, device=device)

        loss = -log_prob * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        episode_reward += reward

        if done:
            break

    print(f"Episode {episode:03d} | Reward: {episode_reward:.1f}")

torch.save(policy.state_dict(), "carplus_model.pt")

print("\n=== TRAINING COMPLETE ===")
print("=== TEST MODE (NO LEARNING) ===\n")

policy.eval()

for test_ep in range(TEST_EPISODES):
    state = torch.tensor(env.reset(), device=device)
    print(f"Test run {test_ep + 1}")

    for _ in range(MAX_STEPS):
        with torch.no_grad():
            mean, _ = policy(state)

        steering = torch.tanh(mean[0]).item()
        throttle = 30

        next_state_np, _, done = env.step(steering, throttle)
        state = torch.tensor(next_state_np, device=device)

        if done:
            break

print("\nDone. You just WATCHED a car learn.\n")
