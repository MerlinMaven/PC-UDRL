import os
import sys
import argparse
import numpy as np
import torch
import imageio
from PIL import Image, ImageDraw, ImageFont

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import make_config
from pc_udrl.utils import get_device, load_checkpoint
from pc_udrl.agents.udrl_agent import UDRLAgent
from pc_udrl.pessimists.quantile import QuantileRegressor
from pc_udrl.pessimists.cvae import CVAEPessimist
from pc_udrl.pessimists.diffusion import CondDiffusion
from pc_udrl.envs.gym_wrappers import make_env
from pc_udrl.data.dataset import OfflineDataset

def build_system(phase):
    cfg = make_config(phase)
    device = get_device(cfg)
    env = make_env(cfg.env_id, cfg.seed)
    
    # Dimensions
    shape = env.observation_space
    state_dim = int(shape[0])
    action_dim = int(env.action_space)
    
    # Models
    agent = UDRLAgent(state_dim, action_dim, discrete=False, hidden_dim=cfg.hidden_dim).to(device)
    
    if cfg.method == "quantile":
        pessimist = QuantileRegressor(state_dim, hidden_dim=cfg.hidden_dim, q=cfg.pessimist_quantile, cfg=cfg)
    elif cfg.method == "cvae":
        pessimist = CVAEPessimist(state_dim, latent_dim=cfg.cvae_latent_dim, hidden_dim=cfg.hidden_dim, cfg=cfg)
    elif cfg.method == "diffusion":
        pessimist = CondDiffusion(state_dim, hidden_dim=cfg.hidden_dim, cfg=cfg, timesteps=100)
    else:
        # Fallback 
        if phase == 2:
             pessimist = QuantileRegressor(state_dim, hidden_dim=cfg.hidden_dim, q=cfg.pessimist_quantile, cfg=cfg)
        elif phase == 3:
             pessimist = CVAEPessimist(state_dim, latent_dim=cfg.cvae_latent_dim, hidden_dim=cfg.hidden_dim, cfg=cfg)
        elif phase == 4:
             pessimist = CondDiffusion(state_dim, hidden_dim=cfg.hidden_dim, cfg=cfg, timesteps=100)
        
    # Load
    run_dir = os.path.join(cfg.runs_dir, f"phase{phase}")
    try:
        # For Phase 4, if agent.pt doesn't exist, use Phase 2 agent
        agent_path = os.path.join(run_dir, "agent.pt")
        if phase == 4 and not os.path.exists(agent_path):
             agent_path = os.path.join(cfg.runs_dir, "phase2", "agent.pt")
             
        load_checkpoint(agent, agent_path, map_location=device)
        load_checkpoint(pessimist, os.path.join(run_dir, "pessimist.pt"), map_location=device)
    except Exception as e:
        print(f"Failed to load models for Phase {phase}: {e}")
        return None
        
    return cfg, env, agent, pessimist, device

def get_action(phase, cfg, env, agent, pessimist, state, steps, target_cmd, device):
    st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    h = torch.tensor([1000 - steps], dtype=torch.float32, device=device)
    
    # Pessimism Logic
    if phase == 2: # Quantile
        cap = pessimist(st)
        cap_val = float(cap.item())
        safe_cmd = min(target_cmd, cap_val)
        cmd_ten = torch.tensor([safe_cmd], dtype=torch.float32, device=device)
        
    elif phase == 3: # CVAE
        cap = pessimist.sample_cap(st, nsamples=16, percentile=cfg.pessimist_quantile)
        cap_val = float(cap.item())
        safe_cmd = min(target_cmd, cap_val)
        cmd_ten = torch.tensor([safe_cmd], dtype=torch.float32, device=device)
        
    elif phase == 4: # Diffusion
        # Diffusion Sample Cap
        # Use lower percentile (e.g. 0.1) to demonstrate safety (Adaptive Braking)
        cap = pessimist.sample_cap(st, nsamples=16, percentile=0.1) 
        cap_val = float(cap.item())
        safe_cmd = min(target_cmd, cap_val)
        cmd_ten = torch.tensor([safe_cmd], dtype=torch.float32, device=device)
        
    a = agent.act(st, h, cmd_ten)
    return a.squeeze(0).detach().cpu().numpy()

def add_label(frame, text):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    # Simple default font
    # font = ImageFont.load_default() 
    # To make it readable without external fonts, we can just draw text
    # or rely on default.
    draw.text((10, 10), text, fill=(255, 255, 255))
    return np.array(img)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--target", type=float, default=195.0)
    args = parser.parse_args()
    
    print("Loading Systems...")
    s2 = build_system(2)
    s3 = build_system(3)
    s4 = build_system(4)
    
    if not s2 or not s3 or not s4:
        print("Failed to load one or more systems. Ensure all phases are trained.")
        return

    envs = [s2[1], s3[1], s4[1]]
    
    print(f"Commanding Target Return: {args.target}")

    all_frames = []
    
    for ep in range(args.episodes):
        obs = [env.reset() for env in envs]
        dones = [False] * 3
        steps = 0
        
        while not all(dones) and steps < 1000:
            actions = []
            
            # Phase 2
            if not dones[0]:
                a2 = get_action(2, s2[0], s2[1], s2[2], s2[3], obs[0], steps, args.target, s2[4])
                o2, _, d2, _ = s2[1].step(a2)
                obs[0] = o2
                dones[0] = d2
            
            # Phase 3
            if not dones[1]:
                a3 = get_action(3, s3[0], s3[1], s3[2], s3[3], obs[1], steps, args.target, s3[4])
                o3, _, d3, _ = s3[1].step(a3)
                obs[1] = o3
                dones[1] = d3
                
            # Phase 4
            if not dones[2]:
                a4 = get_action(4, s4[0], s4[1], s4[2], s4[3], obs[2], steps, args.target, s4[4])
                o4, _, d4, _ = s4[1].step(a4)
                obs[2] = o4
                dones[2] = d4
            
            # Render
            frames = []
            if all(d is not True for d in dones): # Just checking existence
                f2 = envs[0].render()
                f3 = envs[1].render()
                f4 = envs[2].render()
                
                if f2 is not None and f3 is not None and f4 is not None:
                    f2 = add_label(f2, "Phase 2: Quantile")
                    f3 = add_label(f3, "Phase 3: CVAE")
                    f4 = add_label(f4, "Phase 4: Diffusion")
                    combined = np.concatenate([f2, f3, f4], axis=1)
                    all_frames.append(combined)
            
            steps += 1
        print(f"Episode {ep+1} complete.")
        
    # Save
    if all_frames:
        vid_dir = os.path.join(s4[0].runs_dir, "phase4", "videos")
        os.makedirs(vid_dir, exist_ok=True)
        vid_path = os.path.join(vid_dir, "compare_all_phases.mp4")
        imageio.mimsave(vid_path, all_frames, fps=30)
        print(f"Comparison video saved to {vid_path}")

if __name__ == "__main__":
    main()
