import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

plane = p.loadURDF("plane.urdf")
car = p.loadURDF("racecar/racecar.urdf", [0, 0, 0.2])

while True:
    p.stepSimulation()
    time.sleep(1/60)
