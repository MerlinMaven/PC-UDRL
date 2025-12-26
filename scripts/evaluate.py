import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import torch
import imageio
import matplotlib.pyplot as plt
import numpy as np

from pc_udrl.agents.udrl_agent import UDRLAgent
from pc_udrl.pessimists.quantile import QuantileRegressor
# from pc_udrl.pessimists.cvae import CVAEPessimist
from pc_udrl.envs.gridworld import GridWorld
from pc_udrl.envs.gym_wrappers import make_env
from pc_udrl.data.dataset import OfflineDataset
from pc_udrl.utils import get_device, load_checkpoint, set_global_seed
from pc_udrl.utils import Logger
# from pc_udrl.metrics.metrics import normalized_score, cvar, worst_case


def _make_env(cfg):
    if cfg.env_id == "GridWorld":
        return GridWorld(size=5, seed=cfg.seed)
    return make_env(cfg.env_id, cfg.seed)


def _build_models(cfg, env):
    device = get_device(cfg)
    # Infer dimensions from environment
    if cfg.env_id == "GridWorld":
        state_dim = 2
        discrete = True
        action_dim = 4
    else:
        shape = env.observation_space
        state_dim = int(shape[0])
        discrete = bool(getattr(env, "is_discrete", False))
        action_dim = int(env.action_space)

    # Instantiate models
    agent = UDRLAgent(state_dim, action_dim, discrete, hidden_dim=cfg.hidden_dim).to(device)
    if cfg.method == "quantile":
        pessimist = QuantileRegressor(state_dim, hidden_dim=cfg.hidden_dim, q=cfg.pessimist_quantile, cfg=cfg)
    else:
        # pessimist = CVAEPessimist(state_dim, latent_dim=cfg.cvae_latent_dim, hidden_dim=cfg.hidden_dim, cfg=cfg)
        raise ValueError("Only quantile supported in Phase 1 isolated")

    # Load checkpoints
    run_dir = os.path.join(cfg.runs_dir, f"phase{cfg.phase}")
    agent_path = os.path.join(run_dir, "agent.pt")
    pess_path = os.path.join(run_dir, "pessimist.pt")
    load_checkpoint(agent, agent_path, map_location=device)
    load_checkpoint(pessimist, pess_path, map_location=device)
    return agent, pessimist


def evaluate_agent(cfg, use_pessimism: bool = True, capture_video: bool = False, show: bool = False):
    set_global_seed(cfg.seed)
    device = get_device(cfg)
    env = _make_env(cfg)
    agent, pessimist = _build_models(cfg, env)
    dataset = OfflineDataset(cfg)
    target_return = None
    if dataset.exists():
        data = dataset.load()
        target_return = float(torch.max(data["returns"]).item())
    else:
        target_return = 0.0

    returns = []
    gaps = []
    fig = None
    ax = None
    im = None
    if show:
        try:
            plt.ion()
        except Exception:
            show = False
            
    for ep in range(cfg.eval_episodes):
        s = env.reset()
        done = False
        total = 0.0
        steps = 0
        frames = []
        episode_gaps = []
        
        while not done and steps < cfg.max_steps:
            st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Pessimism Logic
            if cfg.method == "quantile":
                cap = pessimist(st)
            # elif cfg.method == "cvae":
            #     cap = pessimist.sample_cap(st, nsamples=16, percentile=cfg.pessimist_quantile)
            else:
                # cap = pessimist.sample_cap(st, nsamples=16, percentile=cfg.pessimist_quantile)
                raise ValueError("Only quantile supported")
            
            cap_val = float(cap.squeeze().detach().cpu().item())
            gap = max(0.0, target_return - cap_val)
            episode_gaps.append(gap)
            
            # Command Generation
            h = torch.tensor([cfg.max_steps - steps], dtype=torch.float32, device=device)
            if use_pessimism:
                cmd_return = torch.tensor([min(target_return, cap_val)], dtype=torch.float32, device=device)
            else:
                cmd_return = torch.tensor([target_return], dtype=torch.float32, device=device)
                
            a = agent.act(st, h, cmd_return)
            a_np = a.squeeze(0).detach().cpu().numpy()
            
            if agent.discrete:
                o2, r, done, _ = env.step(int(a_np))
            else:
                o2, r, done, _ = env.step(a_np)
                
            if capture_video or show:
                frame = env.render()
                if frame is not None:
                    if capture_video: frames.append(frame)
                    if show:
                        if fig is None:
                            fig, ax = plt.subplots(figsize=(5, 5))
                            im = ax.imshow(frame)
                            ax.set_axis_off()
                        else:
                            im.set_data(frame)
                        plt.draw()
                        plt.pause(0.1)
            s = o2
            total += r
            steps += 1
            
        returns.append(total)
        gaps.extend(episode_gaps)
        
        if capture_video and frames:
            mode_str = "pessimistic" if use_pessimism else "standard"
            vid_dir = os.path.join(cfg.runs_dir, f"phase{cfg.phase}", "videos", mode_str)
            os.makedirs(vid_dir, exist_ok=True)
            path = os.path.join(vid_dir, f"eval_ep_{ep + 1}.mp4")
            try:
                with imageio.get_writer(path, fps=30) as writer:
                    for f in frames:
                        writer.append_data(f)
            except Exception as e:
                print(f"Video save failed: {e}")
                
            # Plot Optimization Gap
            if ep == 0 and use_pessimism:
                plt.figure()
                plt.plot(episode_gaps)
                plt.title("Pessimism Gap over Time")
                plt.xlabel("Step")
                plt.ylabel("Gap (Target - Cap)")
                plt.savefig(os.path.join(vid_dir, "gap_analysis.png"))
                plt.close()

    avg_return = sum(returns) / max(1, len(returns))
    avg_gap = sum(gaps) / max(1, len(gaps))
    print(f"[{'Pessimistic' if use_pessimism else 'Standard'}] avg_return: {avg_return:.2f}, avg_gap: {avg_gap:.2f}")
    
    # Log
    mode_str = "pessimist" if use_pessimism else "standard"
    log_dir = os.path.join(cfg.runs_dir, f"phase{cfg.phase}")
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(os.path.join(log_dir, f"eval_{mode_str}.csv"), fieldnames=["avg_return", "avg_pessimism_gap"])
    logger.log({"avg_return": avg_return, "avg_pessimism_gap": avg_gap})
    
    return avg_return, avg_gap


def evaluate_baseline(cfg, algo_name: str, capture_video: bool = False, show: bool = False):
    set_global_seed(cfg.seed)
    device = "cuda" if get_device(cfg).type == "cuda" else "cpu"
    env = _make_env(cfg)
    
    # Load d3rlpy model
    log_dir = os.path.join(cfg.runs_dir, f"phase{cfg.phase}", "baselines", algo_name)
    model_path = os.path.join(log_dir, "model.d3")
    if not os.path.exists(model_path):
        print(f"Baseline {algo_name} not found at {model_path}")
        return -1000.0, 0.0

    import d3rlpy
    # Re-create config to load model structure
    if algo_name == "cql":
        algo = d3rlpy.algos.CQLConfig().create(device=device)
    elif algo_name == "iql":
        algo = d3rlpy.algos.IQLConfig().create(device=device)
    elif algo_name == "td3plusbc":
        algo = d3rlpy.algos.TD3PlusBCConfig().create(device=device)
    
    algo.load_model(model_path)
    
    returns = []
    fig = None
    ax = None
    im = None
    if show:
        try:
            plt.ion()
        except:
            show = False

    for ep in range(cfg.eval_episodes):
        s = env.reset()
        done = False
        total = 0.0
        steps = 0
        frames = []
        
        while not done and steps < cfg.max_steps:
             # d3rlpy predict expects (N, D)
            action = algo.predict(np.array([s]))[0]
            
            o2, r, done, _ = env.step(action)
            
            if capture_video or show:
                frame = env.render()
                if frame is not None:
                    if capture_video: frames.append(frame)
                    if show:
                        if fig is None:
                            fig, ax = plt.subplots(figsize=(5, 5))
                            im = ax.imshow(frame)
                            ax.set_axis_off()
                            ax.set_title(f"{algo_name.upper()} Live")
                        else:
                            im.set_data(frame)
                        plt.draw()
                        plt.pause(0.001)
            s = o2
            total += r
            steps += 1
            
        returns.append(total)
        if capture_video and frames:
            vid_dir = os.path.join(cfg.runs_dir, f"phase{cfg.phase}", "videos", algo_name)
            os.makedirs(vid_dir, exist_ok=True)
            path = os.path.join(vid_dir, f"eval_ep_{ep + 1}.mp4")
            imageio.mimsave(path, frames, fps=30)
            
    avg_return = sum(returns) / max(1, len(returns))
    print(f"[{algo_name.upper()}] avg_return: {avg_return:.2f}")
    if show and fig:
        plt.close(fig)
    return avg_return, 0.0


def compare_modes(cfg):
    print("\n--- Comparative Evaluation ---")
    results = {}
    
    print("1. Standard UDRL")
    r_std, _ = evaluate_agent(cfg, use_pessimism=False, capture_video=True)
    results["Standard"] = r_std
    
    print("2. PC-UDRL (Pessimistic)")
    r_pc, _ = evaluate_agent(cfg, use_pessimism=True, capture_video=True)
    results["PC-UDRL"] = r_pc
    
    # Baselines
    for algo in ["cql", "iql", "td3plusbc"]:
        print(f"Checking {algo}...")
        r_base, _ = evaluate_baseline(cfg, algo, capture_video=True)
        if r_base > -999:
            results[algo.upper()] = r_base
            
    print("\n--- Final Results ---")
    for k, v in results.items():
        print(f"{k:<15}: {v:.2f}")

def evaluate(cfg, capture_video: bool = False, show: bool = False, algo: str = None):
    if algo:
        evaluate_baseline(cfg, algo, capture_video=capture_video, show=show)
    else:
        evaluate_agent(cfg, use_pessimism=True, capture_video=capture_video, show=show)


def visualize_side_by_side(cfg, episodes=3):
    set_global_seed(cfg.seed)
    device = get_device(cfg)
    
    # 1. Init Envs (Identical Seeds)
    if cfg.env_id == "GridWorld":
        env_std = GridWorld(size=5, seed=cfg.seed)
        env_pc = GridWorld(size=5, seed=cfg.seed)
    else:
        env_std = make_env(cfg.env_id, seed=cfg.seed)
        env_pc = make_env(cfg.env_id, seed=cfg.seed)
        
    # 2. Init Models (Shared weights, just different usage)
    agent, pessimist = _build_models(cfg, env_std)
    
    dataset = OfflineDataset(cfg)
    target_return = 0.0
    if dataset.exists():
         data = dataset.load()
         target_return = float(torch.max(data["returns"]).item())
    
    print(f"Starting Side-by-Side Viz for {episodes} episodes...")
    print(f"Target Return: {target_return}")
    
    # Setup Live Viz
    try:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 5))
        im = None
        
        all_frames = []
        
        for ep in range(episodes):
            s_std = env_std.reset()
            s_pc = env_pc.reset()
            
            done_std, done_pc = False, False
            steps = 0
            
            while (not done_std or not done_pc) and steps < cfg.max_steps:
                # --- STANDARD AGENT (Left) ---
                if not done_std:
                    st = torch.tensor(s_std, dtype=torch.float32, device=device).unsqueeze(0)
                    h = torch.tensor([cfg.max_steps - steps], dtype=torch.float32, device=device)
                    cmd = torch.tensor([target_return], dtype=torch.float32, device=device) # Blind command
                    with torch.no_grad():
                        a = agent.act(st, h, cmd)
                    a_np = a.squeeze(0).detach().cpu().numpy()
                    if agent.discrete:
                        s_std, _, done_std, _ = env_std.step(int(a_np))
                    else:
                        s_std, _, done_std, _ = env_std.step(a_np)

                # --- PESSIMISTIC AGENT (Right) ---
                if not done_pc:
                    st = torch.tensor(s_pc, dtype=torch.float32, device=device).unsqueeze(0)
                    h = torch.tensor([cfg.max_steps - steps], dtype=torch.float32, device=device)
                    
                    # Gap Calc
                    cap = pessimist(st)
                    cap_val = float(cap.squeeze().detach().cpu().item())
                    cmd_val = min(target_return, cap_val)
                    cmd = torch.tensor([cmd_val], dtype=torch.float32, device=device)
                    
                    with torch.no_grad():
                        a = agent.act(st, h, cmd)
                    a_np = a.squeeze(0).detach().cpu().numpy()
                    if agent.discrete:
                        s_pc, _, done_pc, _ = env_pc.step(int(a_np))
                    else:
                        s_pc, _, done_pc, _ = env_pc.step(a_np)

                # --- RENDER ---
                frame_std = env_std.render()
                frame_pc = env_pc.render()
                
                # Combine Side-by-Side
                if frame_std is not None and frame_pc is not None:
                    # Ensure same height
                    concat_img = np.concatenate([frame_std, frame_pc], axis=1)
                    
                    # Add to video buffer
                    all_frames.append(concat_img)
                    
                    # Update Window
                    if im is None:
                        im = ax.imshow(concat_img)
                        ax.set_axis_off()
                        ax.set_title("Standard (Left) vs Pessimistic (Right)")
                    else:
                        im.set_data(concat_img)
                    
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(0.01)
                
                steps += 1
        
        plt.close(fig)
        
        # Save Video
        if all_frames:
            vid_dir = os.path.join(cfg.runs_dir, f"phase{cfg.phase}", "videos")
            os.makedirs(vid_dir, exist_ok=True)
            vid_path = os.path.join(vid_dir, "comparison_side_by_side.mp4")
            imageio.mimsave(vid_path, all_frames, fps=30)
            print(f"Comparison video saved to: {vid_path}")

    except Exception as e:
        print(f"Side-by-See Viz failed: {e}")


if __name__ == "__main__":
    from config import make_config
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Run comparison experiment")
    parser.add_argument("--live_compare", action="store_true", help="Run visual side-by-side comparison")
    parser.add_argument("--algo", type=str, help="Baseline algo to evaluate")
    args = parser.parse_args()
    cfg = make_config(args.phase)
    
    if args.live_compare:
        visualize_side_by_side(cfg)
    elif args.compare:
        compare_modes(cfg)
    else:
        evaluate(cfg, capture_video=args.capture_video, show=args.show, algo=args.algo)
