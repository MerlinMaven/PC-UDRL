
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

def compare_agents_ood():
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
    
    print("Starting OOD Comparison...")
    print("Commanding Return: +100.0 (Impossible! Max is +10)")
    
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax_std, ax_pess = axes
    
    for ep in range(2):
        s_std = env_std.reset()
        s_pess = env_pess.reset()
        
        done_std = False
        done_pess = False
        steps = 0
        total_std = 0
        total_pess = 0
        path_std = []
        path_pess = []
        
        while (not done_std or not done_pess) and steps < cfg.max_steps:
            st_std = torch.tensor(s_std, dtype=torch.float32, device=device).unsqueeze(0)
            st_pess = torch.tensor(s_pess, dtype=torch.float32, device=device).unsqueeze(0)
            h = torch.tensor([cfg.max_steps - steps], dtype=torch.float32, device=device)
            
            # OOD COMMAND: +100
            target_ood = 100
                   
            
            # --- Standard Agent Logic ---
            if not done_std:
                # Blindly asks for 100
                cmd_std = torch.tensor([target_ood], dtype=torch.float32, device=device)
                a_std = agent.act(st_std, h, cmd_std)
                act_std = int(a_std.squeeze(0).detach().cpu().numpy())
                s_std, r, done_std, _ = env_std.step(act_std)
                total_std += r
            
            # --- Pessimistic Agent Logic ---
            if not done_pess:
                # Clamps 100 to Pessimist Cap
                cap = pessimist(st_pess)
                cap_val = float(cap.squeeze().detach().cpu().item())
                
                # IMPORTANT: Cap should vary. Near goal -> +10. Far from goal -> lower?
                # Ideally, min(100, cap) gives the best REALISTIC return.
                cmd_pess = torch.tensor([min(target_ood, cap_val)], dtype=torch.float32, device=device)
                
                a_pess = agent.act(st_pess, h, cmd_pess)
                act_pess = int(a_pess.squeeze(0).detach().cpu().numpy())
                s_pess, r, done_pess, _ = env_pess.step(act_pess)
                total_pess += r
                
                # Debug print for first step
                if steps == 0:
                    print(f"[Ep {ep}] Pessimist Cap: {cap_val:.2f} -> Command: {min(target_ood, cap_val):.2f}")
            
            # Track Positions
            if not done_std:
                path_std.append((int(s_std[0]), int(s_std[1])))
            if not done_pess:
                path_pess.append((int(s_pess[0]), int(s_pess[1])))

            # --- Visualization ---
            frame_std = env_std.render()
            frame_pess = env_pess.render()
            
            ax_std.clear()
            ax_std.imshow(frame_std)
            ax_std.set_title(f"Standard UDRL (Cmd={target_ood})\nReward: {total_std:.1f}")
            ax_std.axis('off')
            
            ax_pess.clear()
            ax_pess.imshow(frame_pess)
            ax_pess.set_title(f"PC-UDRL (Cmd=min({target_ood}, Cap))\nReward: {total_pess:.1f}")
            ax_pess.axis('off')
            
            plt.draw()
            plt.pause(0.1)
            steps += 1
            
        print(f"Episode {ep+1} Finished.")
        print(f"  Standard Return: {total_std} | Path: {path_std}")
        print(f"  Pessimist Return: {total_pess} | Path: {path_pess}")
        print(f"  Goal: {env_std.goal}")
        plt.pause(1.0)

    plt.ioff()
    plt.close()

if __name__ == "__main__":
    compare_agents_ood()
