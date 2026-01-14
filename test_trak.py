import pybullet as p
import pybullet_data
import numpy as np
import math
import time

# ================= TRACK PARAMETERS =================
TRACK_WIDTH = 3.5      # distance from center to wall
WALL_THICKNESS = 0.15  # very slim wall (like image)
WALL_HEIGHT = 0.6
SEGMENTS = 600         # higher = smoother & continuous

# ================= TRACK SHAPE (LIKE IMAGE) =================
def track_curve(t):
    """
    Parametric curve (0..1) that produces
    a complex F1-style track like the image
    """
    angle = 2 * math.pi * t

    x = (
        35 * math.cos(angle)
        + 10 * math.cos(2 * angle)
        - 6 * math.sin(3 * angle)
    )

    y = (
        20 * math.sin(angle)
        + 8 * math.sin(2 * angle)
    )

    return np.array([x, y])

# ================= WALL CREATION =================
def create_wall_segment(pos, yaw, length):
    col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[length / 2, WALL_THICKNESS, WALL_HEIGHT]
    )
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[length / 2, WALL_THICKNESS, WALL_HEIGHT],
        rgbaColor=[0.25, 0.25, 0.25, 1]
    )
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[pos[0], pos[1], WALL_HEIGHT],
        baseOrientation=p.getQuaternionFromEuler([0, 0, yaw])
    )

# ================= BUILD TRACK =================
def build_track():
    for i in range(SEGMENTS):
        t1 = i / SEGMENTS
        t2 = (i + 1) / SEGMENTS

        p1 = track_curve(t1)
        p2 = track_curve(t2)

        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length == 0:
            continue

        direction /= length
        normal = np.array([-direction[1], direction[0]])
        yaw = math.atan2(direction[1], direction[0])

        center = (p1 + p2) / 2

        left = center + normal * TRACK_WIDTH
        right = center - normal * TRACK_WIDTH

        create_wall_segment(left, yaw, length)
        create_wall_segment(right, yaw, length)

# ================= MAIN =================
p.connect(p.GUI)
p.setGravity(0, 0, -9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.loadURDF("plane.urdf")
build_track()

# Car
car = p.loadURDF("racecar/racecar.urdf", [30, 0, 0.4])

# Camera
p.resetDebugVisualizerCamera(
    cameraDistance=120,
    cameraYaw=45,
    cameraPitch=-70,
    cameraTargetPosition=[0, 0, 0]
)

# Simulation loop
while True:
    p.stepSimulation()
    time.sleep(1 / 60)
