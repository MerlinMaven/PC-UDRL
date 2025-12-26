
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pc_udrl.envs.gridworld import GridWorld
from pc_udrl.data.dataset import OfflineDataset
from config import make_config
import torch

def debug_phase1():
    print("--- 1. Checking Map Layout ---")
    env = GridWorld(size=5, seed=0, fixed_map=True)
    print("Walls:", env.walls)
    print("Traps:", env.traps)
    print("Agent Start:", env.agent)
    print("Goal:", env.goal)
    
    # Simple check if goal is surrounded
    # (Manual check via print output or simple BFS if needed, but visual inspection of coords is faster)
    
    print("\n--- 2. Checking Dataset Statistics ---")
    cfg = make_config(1)
    dataset = OfflineDataset(cfg)
    if not dataset.exists():
        print("Dataset not found!")
        return
    
    data = dataset.load()
    returns = data["returns"]
    max_ret = torch.max(returns).item()
    mean_ret = torch.mean(returns).item()
    success_count = (returns > 5.0).sum().item() # Success is +10 (minus steps)
    
    print(f"Total Episodes: {len(returns)}")
    print(f"Max Return: {max_ret}")
    print(f"Mean Return: {mean_ret}")
    print(f"Successful Episodes (> 5.0 return): {success_count}")

if __name__ == "__main__":
    debug_phase1()
