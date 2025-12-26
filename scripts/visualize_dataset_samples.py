
import os
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
import imageio

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pc_udrl.utils import get_device, set_global_seed
from pc_udrl.envs.gridworld import GridWorld
from config import make_config

def visualize_dataset():
    cfg = make_config(1)
    dataset_path = os.path.join(cfg.dataset_dir, f"{cfg.env_id}.pt")
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    print(f"Loading dataset from {dataset_path}...")
    data = torch.load(dataset_path)
    
    # Reconstruct Episodes
    obs = data["obs"].numpy()
    rewards = data["rewards"].numpy()
    dones = data["dones"].numpy()
    returns = data["returns"].numpy() # This is RTG, not episode return. 
    # Actually, OfflineDataset calculates RTG. Episode return is R(t=0).
    # We need to split by 'dones'.
    
    episodes = []
    current_ep_return = 0
    current_ep_obs = []
    start_idx = 0
    
    for i in range(len(dones)):
        current_ep_obs.append(obs[i])
        current_ep_return += rewards[i]
        
        if dones[i] > 0.5: # Terminal
            # For RTG check: The existing 'returns' tensor, at the start index of the episode,
            # should roughly equal the total discounted return. 
            # But let's just use the sum of rewards for histogram (Undiscounted Return often used for 'success' metric).
            episodes.append({
                "return": current_ep_return,
                "length": len(current_ep_obs),
                "obs": current_ep_obs
            })
            current_ep_return = 0
            current_ep_obs = []
            start_idx = i + 1
            
    print(f"Total Episodes found: {len(episodes)}")
    
    # 1. Plot Histogram of Returns
    ep_returns = [ep["return"] for ep in episodes]
    plt.figure(figsize=(10, 6))
    plt.hist(ep_returns, bins=50, color='skyblue', edgecolor='black')
    plt.title("Distribution of Episode Returns (Mixed Dataset)")
    plt.xlabel("Return")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    
    hist_path = os.path.join(cfg.runs_dir, f"phase{cfg.phase}", "plots", "dataset_histogram.png")
    os.makedirs(os.path.dirname(hist_path), exist_ok=True)
    plt.savefig(hist_path)
    print(f"Histogram saved to {hist_path}")
    
    # 2. Render Samples
    # Find one BEST return (Expert) and one MEDIAN return (Random)
    sorted_eps = sorted(episodes, key=lambda x: x["return"])
    
    best_ep = sorted_eps[-1]
    median_ep = sorted_eps[len(sorted_eps)//2]
    worst_ep = sorted_eps[0]
    
    print(f"Best Return: {best_ep['return']:.2f} (Length: {best_ep['length']})")
    print(f"Median Return: {median_ep['return']:.2f} (Length: {median_ep['length']})")
    print(f"Worst Return: {worst_ep['return']:.2f} (Length: {worst_ep['length']})")
    
    # Helper to render
    env = GridWorld(size=5, seed=0, fixed_map=True) # Use fixed map for visualization consistency
    
    def render_episode(ep, name):
        frames = []
        # We can't perfectly replay if dynamics are stochastic, but GridWorld is deterministic given state/action.
        # However, we only have states. We can just set the agent position and render.
        # GridWorld state is just (y, x).
        
        for ob in ep["obs"]:
            env.agent = ob.astype(float) # Set state directly
            frames.append(env.render())
            
        vid_path = os.path.join(cfg.runs_dir, f"phase{cfg.phase}", "plots", f"dataset_sample_{name}.mp4")
        imageio.mimsave(vid_path, frames, fps=5)
        print(f"Video saved to {vid_path}")
        return vid_path

    render_episode(best_ep, "expert_best")
    render_episode(median_ep, "random_median")

if __name__ == "__main__":
    visualize_dataset()
