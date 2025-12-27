
import sys
import os
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

# Ensure src is in path to import GridWorld
sys.path.insert(0, os.path.abspath('src'))
try:
    from pc_udrl.envs.gridworld import GridWorld
except ImportError:
    # Fallback if running from a different relative path or src structure differs
    sys.path.insert(0, os.path.abspath('.'))
    from src.pc_udrl.envs.gridworld import GridWorld

def save_image(img_array, path):
    if img_array is None:
        print(f"Error: Image array for {path} is None")
        return
    plt.imsave(path, img_array)
    print(f"Saved {path}")

def generate_gridworld_image():
    print("Generating GridWorld image...")
    # Matches Config (Phase 1): size=5, seed=0
    env = GridWorld(size=5, seed=0) 
    env.reset()
    img = env.render(cell_px=64)
    save_image(img, 'assets/gridworld_init.png')

def generate_lunarlander_image():
    print("Generating LunarLander image...")
    # Try different versions just in case
    pkgs = ["LunarLanderContinuous-v3", "LunarLander-v3", "LunarLanderContinuous-v2", "LunarLander-v2"] 
    env = None
    for pkg in pkgs:
        try:
            env = gym.make(pkg, render_mode="rgb_array")
            break
        except Exception as e:
            print(f"Could not load {pkg}: {e}")
    
    if env is None:
        print("Failed to load LunarLander env.")
        return

    env.reset(seed=42)
    
    # Simulate freefall for 60 steps to bring the lander into center view
    print("Simulating freefall...")
    action = 0 if isinstance(env.action_space, gym.spaces.Discrete) else np.array([0.0, 0.0])
    for _ in range(60):
        env.step(action)
        
    img = env.render()
    save_image(img, 'assets/lunarlander_init.png')
    env.close()

if __name__ == "__main__":
    if not os.path.exists('assets'):
        os.makedirs('assets')
    generate_gridworld_image()
    generate_lunarlander_image()
