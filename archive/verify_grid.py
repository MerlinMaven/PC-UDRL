
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pc_udrl.envs.gridworld import GridWorld
import numpy as np

env = GridWorld(size=5, seed=0)
print("Episode 1 Walls:", env.walls)
env.reset()
print("Episode 2 Walls:", env.walls)
env.reset()
print("Episode 3 Walls:", env.walls)
