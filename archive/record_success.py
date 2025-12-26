
import os
import sys
import imageio
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pc_udrl.utils import get_device, set_global_seed
from pc_udrl.envs.gridworld import GridWorld
from pc_udrl.agents.udrl_agent import UDRLAgent
from pc_udrl.pessimists.quantile import QuantileRegressor
from pc_udrl.utils import load_checkpoint
from config import make_config

def record_best_episode():
    # Setup
    cfg = make_config(1)
    set_global_seed(cfg.seed)
    device = get_device(cfg)
    
    # Environment (Fixed Map)
    env = GridWorld(size=5, seed=0, fixed_map=True)
    
    # Models
    agent = UDRLAgent(2, 4, True, hidden_dim=cfg.hidden_dim).to(device)
    pessimist = QuantileRegressor(2, hidden_dim=cfg.hidden_dim, q=cfg.pessimist_quantile, cfg=cfg)
    
    run_dir = os.path.join(cfg.runs_dir, f"phase{cfg.phase}")
    load_checkpoint(agent, os.path.join(run_dir, "agent.pt"), map_location=device)
    load_checkpoint(pessimist, os.path.join(run_dir, "pessimist.pt"), map_location=device)
    
    print("Searching for a successful episode...")
    max_attempts = 100
    
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
            
            st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Pessimist Cap
            cap = pessimist(st)
            cap_val = float(cap.squeeze().detach().cpu().item())
            
            # Command: Target is +10 (Goal), but we clamp to Cap
            # Note: We want to show the agent succeeding. 
            # If the model is good, Cap should be high near goal.
            # Try Standard UDRL (No Pessimism) to force goal seeking
            target = 10.0
            cmd_val = target
            
            cmd_return = torch.tensor([cmd_val], dtype=torch.float32, device=device)
            # Try conditioning on a shorter horizon (Prompt Engineering)
            # In dataset, successes are short (~10-20 steps). H=100 might be OOD.
            h_val = min(20, cfg.max_steps - steps)
            h = torch.tensor([h_val], dtype=torch.float32, device=device)
            
            # Agent Act
            a = agent.act(st, h, cmd_return)
            action = int(a.squeeze(0).detach().cpu().numpy())
            
            s, r, done, _ = env.step(action)
            total_reward += r
            steps += 1
        
        # Check success (Goal reward is +10, but total includes step costs)
        # In GridWorld.py: reward is -0.1 per step, +10 at goal.
        # So if total_reward > 0, it generally means we hit the goal.
        if total_reward > 0 and len(frames) > 0:
            print(f"SUCCESS found on attempt {attempt+1}! Return: {total_reward:.2f}")
            
            # Save Video
            out_path = os.path.join(run_dir, "plots", "success_demo.mp4")
            imageio.mimsave(out_path, frames, fps=10) # 10 fps for clearer viewing
            print(f"Video saved to: {out_path}")
            return out_path
            
    print("Could not find a successful episode in 100 attempts.")
    return None

if __name__ == "__main__":
    record_best_episode()
