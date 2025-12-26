import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import make_config
from pc_udrl.data.dataset import OfflineDataset

def inspect_data(phase: int):
    cfg = make_config(phase)
    dataset = OfflineDataset(cfg)
    
    if not dataset.exists():
        print(f"No dataset found for Phase {phase}")
        return

    print(f"Loading dataset for Phase {phase}...")
    data = dataset.load()
    
    obs = data["obs"].numpy()
    rewards = data["rewards"].numpy()
    returns = data["returns"].numpy()
    
    print(f"Dataset Size: {len(obs)} steps")
    print(f"Max Return: {returns.max():.2f}")
    print(f"Mean Reward: {rewards.mean():.4f}")

    # Success Analysis
    # In GridWorld, reaching goal gives +10.0 reward at the last step
    # We can count how many episodes have a reward of 10.0
    ep_rewards = data["rewards"].numpy()
    success_count = np.sum(ep_rewards == 10.0)
    total_episodes = 200 # We know this from config, or could infer
    success_rate = (success_count / total_episodes) * 100

    print("-" * 30)
    print(f"ROBUST ANALYSIS:")
    print(f"Episodes Reaching Goal: {success_count} / {total_episodes}")
    print(f"Success Rate (Random):  {success_rate:.2f}%")
    print("-" * 30)

    print("-" * 30)
    print("DATA SAMPLE (What the Agent Sees):")
    print(f"{'Idx':<6} | {'Horizon':<8} | {'Return (TG)':<12} | {'State':<15} | {'Action':<6} | {'Reward':<6}")
    print("-" * 65)
    
    # Randomly sample 10 points
    indices = np.random.choice(len(obs), 10, replace=False)
    for idx in indices:
        s_str = f"[{obs[idx][0]:.1f}, {obs[idx][1]:.1f}]"
        print(f"{idx:<6} | {data['horizons'][idx]:<8.1f} | {data['returns'][idx]:<12.1f} | {s_str:<15} | {data['actions'][idx]:<6} | {data['rewards'][idx]:<6.1f}")
    print("-" * 65)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "graphs", "inspection")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Component Analysis (GridWorld Specific)
    if cfg.env_id == "GridWorld":
        # Obs are (y, x) normalized? No, GridWorld is usually raw coords in this impl?
        # Let's check ranges.
        y_coords = obs[:, 0]
        x_coords = obs[:, 1]
        
        plt.figure(figsize=(8, 6))
        # 2D Histogram / Heatmap
        plt.hist2d(x_coords, y_coords, bins=5, range=[[0, 5], [0, 5]], cmap="viridis", cmin=1)
        plt.colorbar(label="Visit Count")
        plt.title(f"State Coverage Heatmap ({len(obs)} steps)")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.gca().invert_yaxis() # GridWorld usually (0,0) top-left
        save_path = os.path.join(output_dir, "state_coverage.png")
        plt.savefig(save_path)
        print(f"Saved coverage map to {save_path}")
        plt.close()

    # 2. Return Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(returns, bins=20, kde=True)
    plt.title("Return Distribution (Monte Carlo)")
    plt.xlabel("Return")
    plt.ylabel("Count")
    save_path = os.path.join(output_dir, "return_dist.png")
    plt.savefig(save_path)
    print(f"Saved return distribution to {save_path}")
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1)
    args = parser.parse_args()
    inspect_data(args.phase)
    sys.exit(0)
