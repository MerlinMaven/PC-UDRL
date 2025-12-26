
import os
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pc_udrl.utils import get_device, set_global_seed, load_checkpoint
from pc_udrl.envs.gridworld import GridWorld
from pc_udrl.agents.udrl_agent import UDRLAgent
from pc_udrl.pessimists.quantile import QuantileRegressor
from config import make_config

def compare_agents():
    cfg = make_config(1)
    set_global_seed(cfg.seed)
    device = get_device(cfg)
    
    # Create two identical environments (Fixed Map)
    env_std = GridWorld(size=5, seed=0, fixed_map=True)
    env_pess = GridWorld(size=5, seed=0, fixed_map=True)
    
    # Load Models
    agent = UDRLAgent(2, 4, True, hidden_dim=cfg.hidden_dim).to(device)
    pessimist = QuantileRegressor(2, hidden_dim=cfg.hidden_dim, q=cfg.pessimist_quantile, cfg=cfg)
    
    run_dir = os.path.join(cfg.runs_dir, f"phase{cfg.phase}")
    load_checkpoint(agent, os.path.join(run_dir, "agent.pt"), map_location=device)
    load_checkpoint(pessimist, os.path.join(run_dir, "pessimist.pt"), map_location=device)
    
    print("Starting Side-by-Side Comparison...")
    print("Left: Standard UDRL (Target=10.0)")
    print("Right: PC-UDRL (Target=min(10.0, Pessimist))")
    
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax_std, ax_pess = axes
    
    for ep in range(5): # Run 5 comparison episodes
        s_std = env_std.reset()
        s_pess = env_pess.reset()
        
        done_std = False
        done_pess = False
        
        steps = 0
        total_std = 0
        total_pess = 0
        
        while (not done_std or not done_pess) and steps < cfg.max_steps:
            # Prepare Inputs
            st_std = torch.tensor(s_std, dtype=torch.float32, device=device).unsqueeze(0)
            st_pess = torch.tensor(s_pess, dtype=torch.float32, device=device).unsqueeze(0)
            
            h = torch.tensor([cfg.max_steps - steps], dtype=torch.float32, device=device) # Same horizon
            
            # --- Standard Agent Logic ---
            if not done_std:
                # Naive Command: Always ask for +10
                cmd_std = torch.tensor([10.0], dtype=torch.float32, device=device)
                a_std = agent.act(st_std, h, cmd_std)
                act_std = int(a_std.squeeze(0).detach().cpu().numpy())
                s_std, r, done_std, _ = env_std.step(act_std)
                total_std += r
            
            # --- Pessimistic Agent Logic ---
            if not done_pess:
                # Pessimistic Command
                cap = pessimist(st_pess)
                cap_val = float(cap.squeeze().detach().cpu().item())
                cmd_pess = torch.tensor([min(10.0, cap_val)], dtype=torch.float32, device=device)
                
                a_pess = agent.act(st_pess, h, cmd_pess)
                act_pess = int(a_pess.squeeze(0).detach().cpu().numpy())
                s_pess, r, done_pess, _ = env_pess.step(act_pess)
                total_pess += r
            
            # --- Visualization ---
            frame_std = env_std.render()
            frame_pess = env_pess.render()
            
            ax_std.clear()
            ax_std.imshow(frame_std)
            ax_std.set_title(f"Standard UDRL\nReward: {total_std:.1f}")
            ax_std.axis('off')
            
            ax_pess.clear()
            ax_pess.imshow(frame_pess)
            ax_pess.set_title(f"PC-UDRL (Pessimistic)\nReward: {total_pess:.1f}")
            ax_pess.axis('off')
            
            plt.draw()
            plt.pause(0.1) # Slow down for viewing
            
            steps += 1
            
        print(f"Episode {ep+1} Finished. Std Return: {total_std}, Pess Return: {total_pess}")
        plt.pause(1.0) # Pause between episodes

    plt.ioff()
    plt.close()

if __name__ == "__main__":
    compare_agents()
