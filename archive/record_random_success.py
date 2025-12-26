
import os
import sys
import imageio
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pc_udrl.envs.gridworld import GridWorld
from config import make_config

def record_random_success():
    cfg = make_config(1)
    env = GridWorld(size=5, seed=0, fixed_map=True)
    
    print("Searching for a random successful episode...")
    max_attempts = 10000
    
    for attempt in range(max_attempts):
        s = env.reset()
        done = False
        steps = 0
        frames = []
        total_reward = 0.0
        
        while not done and steps < cfg.max_steps:
            # Render frame
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            
            # Random Action
            action = np.random.randint(0, 4)
            
            s, r, done, _ = env.step(action)
            total_reward += r
            steps += 1
        
        if total_reward > 0: # Positive return means Goal reached reasonably fast
            print(f"SUCCESS found on attempt {attempt+1}! Return: {total_reward:.2f}")
            
            # Add final frame
            frames.append(env.render())
            
            # Save Video
            run_dir = os.path.join(cfg.runs_dir, f"phase{cfg.phase}")
            os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
            out_path = os.path.join(run_dir, "plots", "success_demo.mp4")
            imageio.mimsave(out_path, frames, fps=10)
            print(f"Video saved to: {out_path}")
            return out_path
            
    print("Could not find a successful episode.")
    return None

if __name__ == "__main__":
    record_random_success()
