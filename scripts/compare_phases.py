import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import make_config
from pc_udrl.utils import get_device, load_checkpoint, set_global_seed
from pc_udrl.agents.udrl_agent import UDRLAgent
from pc_udrl.pessimists.quantile import QuantileRegressor
from pc_udrl.pessimists.cvae import CVAEPessimist
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
    else:
        # Fallback if phase 3 config says cvae but we want to be sure
        if phase == 2:
             pessimist = QuantileRegressor(state_dim, hidden_dim=cfg.hidden_dim, q=cfg.pessimist_quantile, cfg=cfg)
        elif phase == 3:
             pessimist = CVAEPessimist(state_dim, latent_dim=cfg.cvae_latent_dim, hidden_dim=cfg.hidden_dim, cfg=cfg)
        
    # Load
    run_dir = os.path.join(cfg.runs_dir, f"phase{phase}")
    try:
        load_checkpoint(agent, os.path.join(run_dir, "agent.pt"), map_location=device)
        load_checkpoint(pessimist, os.path.join(run_dir, "pessimist.pt"), map_location=device)
    except Exception as e:
        print(f"Failed to load models for Phase {phase}: {e}")
        return None
    
    # Determine Target Return (Max from dataset)
    dataset = OfflineDataset(cfg)
    if dataset.exists():
        data = dataset.load()
        target = float(torch.max(data["returns"]).item())
    else:
        target = 200.0 
        
    return cfg, env, agent, pessimist, target, device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    
    print("Loading Phase 2 (Quantile)...")
    sys2 = build_system(2)
    if not sys2: return
    cfg2, env2, agent2, pess2, target2, dev2 = sys2
    
    print("Loading Phase 3 (CVAE)...")
    sys3 = build_system(3)
    if not sys3: return
    cfg3, env3, agent3, pess3, target3, dev3 = sys3
    
    target_cmd = max(target2, target3)
    print(f"Commanding Target Return: {target_cmd:.2f}")

    # Viz Loop
    try:
        # plt.ion()
        # fig, ax = plt.subplots(figsize=(12, 6))
        # im = None
        all_frames = []
        
        print("Starting simulation loop...")
        for ep in range(args.episodes):
            s2 = env2.reset()
            s3 = env3.reset()
            done2, done3 = False, False
            steps = 0
            
            while (not done2 or not done3) and steps < 1000:
                # --- PHASE 2 (Quantile) ---
                if not done2:
                    st = torch.tensor(s2, dtype=torch.float32, device=dev2).unsqueeze(0)
                    h = torch.tensor([1000 - steps], dtype=torch.float32, device=dev2)
                    
                    cap = pess2(st)
                    cap_val = float(cap.item())
                    safe_cmd = min(target_cmd, cap_val)
                    
                    cmd_ten = torch.tensor([safe_cmd], dtype=torch.float32, device=dev2)
                    a = agent2.act(st, h, cmd_ten)
                    a_np = a.squeeze(0).detach().cpu().numpy()
                    s2, _, done2, _ = env2.step(a_np)

                # --- PHASE 3 (CVAE) ---
                if not done3:
                    st = torch.tensor(s3, dtype=torch.float32, device=dev3).unsqueeze(0)
                    h = torch.tensor([1000 - steps], dtype=torch.float32, device=dev3)
                    
                    # CVAE sample
                    cap = pess3.sample_cap(st, nsamples=16, percentile=cfg3.pessimist_quantile)
                    cap_val = float(cap.item())
                    safe_cmd = min(target_cmd, cap_val)
                    
                    cmd_ten = torch.tensor([safe_cmd], dtype=torch.float32, device=dev3)
                    a = agent3.act(st, h, cmd_ten)
                    a_np = a.squeeze(0).detach().cpu().numpy()
                    s3, _, done3, _ = env3.step(a_np)
                    
                # --- RENDER ---
                f2 = env2.render()
                f3 = env3.render()
                
                if f2 is not None and f3 is not None:
                    # Concatenate
                    # Check shape
                    if f2.shape != f3.shape:
                        # Resize f3 to f2?
                        pass
                    
                    combined = np.concatenate([f2, f3], axis=1)
                    all_frames.append(combined)
                
                steps += 1
            print(f"Episode {ep+1} complete.")
                
        # Save
        if all_frames:
            vid_dir = os.path.join(cfg3.runs_dir, "phase3", "videos")
            os.makedirs(vid_dir, exist_ok=True)
            vid_path = os.path.join(vid_dir, "comparison_p2_vs_p3.mp4")
            imageio.mimsave(vid_path, all_frames, fps=30)
            print(f"Comparison video saved to {vid_path}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
