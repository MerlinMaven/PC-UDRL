
import os
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
import imageio

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pc_udrl.utils import get_device, set_global_seed, load_checkpoint
from pc_udrl.envs.gridworld import GridWorld
from pc_udrl.agents.udrl_agent import UDRLAgent
from pc_udrl.pessimists.quantile import QuantileRegressor
from config import make_config

def load_models(run_dir, cfg, device):
    """Loads Agent and Pessimist from a specific run directory."""
    agent = UDRLAgent(2, 4, True, hidden_dim=cfg.hidden_dim).to(device)
    pessimist = QuantileRegressor(2, hidden_dim=cfg.hidden_dim, q=cfg.pessimist_quantile, cfg=cfg)
    
    agent_path = os.path.join(run_dir, "agent.pt")
    pess_path = os.path.join(run_dir, "pessimist.pt")
    
    if not os.path.exists(agent_path) or not os.path.exists(pess_path):
        print(f"Warning: Could not find models in {run_dir}")
        return None, None
        
    load_checkpoint(agent, agent_path, map_location=device)
    load_checkpoint(pessimist, pess_path, map_location=device)
    return agent, pessimist

def benchmark_viz():
    cfg = make_config(1)
    set_global_seed(cfg.seed)
    device = get_device(cfg)
    
    # GridWorld Fixed Map
    env = GridWorld(size=5, seed=0, fixed_map=True)
    goal_pos = env.goal
    
    # Load Models for Experiment A (Random)
    print("Loading Experiment A (Random)...")
    agent_rand, pess_rand = load_models(os.path.join("runs", "phase1_random"), cfg, device)
    
    # Load Models for Experiment B (Mixed)
    print("Loading Experiment B (Mixed)...")
    agent_mix, pess_mix = load_models(os.path.join("runs", "phase1_mixed"), cfg, device)
    
    if agent_rand is None or agent_mix is None:
        print("Models not found. Ensure run_benchmark.py has finished.")
        return

    # Configuration for 2x2 Grid
    # Row 0: Random Dataset, Row 1: Mixed Dataset
    # Col 0: Standard Agent, Col 1: Pessimistic Agent
    models = [
        [("Random / Std", agent_rand, None), ("Random / PC-UDRL", agent_rand, pess_rand)],
        [("Mixed / Std", agent_mix, None),   ("Mixed / PC-UDRL", agent_mix, pess_mix)]
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    video_frames = []
    
    # Reset tracking
    states = [[env.reset() for _ in range(2)] for _ in range(2)]
    dones = [[False for _ in range(2)] for _ in range(2)]
    returns = [[0.0 for _ in range(2)] for _ in range(2)]
    paths = [[[] for _ in range(2)] for _ in range(2)]
    
    target_ood = 100.0 # IMPOSSIBLE COMMAND
    
    for t in range(cfg.max_steps):
        # Gather frames for this step
        step_frames = [[None, None], [None, None]]
        
        # Render current state for all 4 subplots
        for r in range(2):
            for c in range(2):
                # Set env to this agent's state
                env.agent = np.array(states[r][c], dtype=float)
                img = env.render()
                step_frames[r][c] = img
                
                # Plot
                axes[r][c].clear()
                axes[r][c].imshow(img)
                
                name, ag, pess = models[r][c]
                status = "DONE" if dones[r][c] else "ACTING"
                axes[r][c].set_title(f"{name}\nReturn: {returns[r][c]:.1f} | {status}", fontsize=10)
                axes[r][c].axis('off')
        
        # Capture figure for video
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        video_frames.append(image)
        
        # Step Logic
        for r in range(2):
            for c in range(2):
                if dones[r][c]: continue
                
                s = states[r][c]
                st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                h = torch.tensor([cfg.max_steps - t], dtype=torch.float32, device=device)
                
                name, ag, pess = models[r][c]
                
                # Command Logic
                if pess is not None:
                    # PC-UDRL
                    cap = pess(st).squeeze().detach().cpu().item()
                    cmd_val = min(target_ood, cap)
                else:
                    # Standard
                    cmd_val = target_ood
                
                cmd = torch.tensor([cmd_val], dtype=torch.float32, device=device)
                
                # Act
                with torch.no_grad():
                    a = ag.act(st, h, cmd)
                    action = int(a.squeeze(0).cpu().numpy())
                
                # Step (Need to ensure we don't mess up common env, so we use logic manually or reset env)
                # Actually GridWorld.step updates self.agent. 
                # So we must set env.agent before stepping.
                env.agent = np.array(s, dtype=float)
                o2, rew, done, _ = env.step(action)
                
                states[r][c] = o2
                dones[r][c] = done
                returns[r][c] += rew
                paths[r][c].append((int(o2[0]), int(o2[1])))
                
        # If all done, break
        if all(all(row) for row in dones):
            break
            
    # Save Video
    vid_path = os.path.join("runs", "benchmark_2x2.mp4")
    imageio.mimsave(vid_path, video_frames, fps=5)
    print(f"Benchmark video saved to {vid_path}")
    
    # Save Final Frame
    plt.savefig(os.path.join("runs", "benchmark_final.png"))
    print("Final frame saved.")
    
    # Print Results Matrix
    print("\nBenchmark Results (Return / OOD Robustness):")
    print(f"{'Data':<10} | {'Standard Agent':<20} | {'PC-UDRL Agent':<20}")
    print("-" * 60)
    print(f"{'Random':<10} | {returns[0][0]:<20.1f} | {returns[0][1]:<20.1f}")
    print(f"{'Mixed':<10} | {returns[1][0]:<20.1f} | {returns[1][1]:<20.1f}")

if __name__ == "__main__":
    benchmark_viz()
