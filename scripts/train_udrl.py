import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pc_udrl.agents.udrl_agent import UDRLAgent
from pc_udrl.utils import Logger, save_checkpoint, get_device, set_global_seed


def train_udrl(cfg, dataset):
    set_global_seed(cfg.seed)
    device = get_device(cfg)
    if dataset.data is None:
        dataset.load()
    train_ds, val_ds = dataset.split_train_val(0.2)
    pin = device.type in ("cuda", "mps")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=pin)
    actions = dataset.data["actions"]
    state_dim = dataset.data["obs"].shape[1]
    discrete = actions.ndim == 1 and actions.dtype in (torch.int64, torch.int32)
    action_dim = int(actions.max().item()) + 1 if discrete else actions.shape[1] if actions.ndim > 1 else 1
    agent = UDRLAgent(state_dim, action_dim, discrete, hidden_dim=cfg.hidden_dim).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr_agent)
    epochs = getattr(cfg, "epochs", 20)
    log_dir = os.path.join(cfg.runs_dir, f"phase{cfg.phase}")
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(os.path.join(log_dir, "agent.csv"), fieldnames=["epoch", "train_loss", "val_loss"], overwrite=True)
    # Setup Eval Logger
    eval_csv_path = os.path.join(log_dir, "eval.csv")
    eval_logger = Logger(eval_csv_path, fieldnames=["epoch", "avg_return", "avg_pessimism_gap"], overwrite=True)
    
    # Import for evaluation inside function to avoid circular imports
    from scripts.evaluate import evaluate_agent
    
    # Live Rendering Setup
    render_interval = getattr(cfg, "render_interval", 1) if getattr(cfg, "render", False) else 0

    # Load Pessimist if available (for gap analysis)
    pessimist = None
    if cfg.method == "quantile":
        from pc_udrl.pessimists.quantile import QuantileRegressor
        if discrete:
             # GridWorld state dim eq 2?
             pass 
        # State dim logic
        if cfg.env_id == "GridWorld":
            s_dim = 2
        else:
            # We don't have env instance here easily to check dim, but usually it's passed or known.
            # Let's rely on dataset dims.
            s_dim = state_dim

        try:
            pessimist = QuantileRegressor(s_dim, hidden_dim=cfg.hidden_dim, q=cfg.pessimist_quantile, cfg=cfg).to(device)
            pess_path = os.path.join(log_dir, "pessimist.pt")
            if os.path.exists(pess_path):
                from pc_udrl.utils import load_checkpoint
                load_checkpoint(pessimist, pess_path, map_location=device)
                print("Loaded pessimist for evaluation.")
            else:
                print("Pessimist checkpoint not found, gap will be 0.")
                pessimist = None
        except Exception as e:
            print(f"Failed to load pessimist: {e}")
            pessimist = None

    # Setup Live Viz Objects
    viz_env = None
    fig, ax, im = None, None, None
    if render_interval > 0:
        try:
            import matplotlib.pyplot as plt
            plt.ion()
            fig, ax = plt.subplots(figsize=(5, 5))
            plt.show(block=False)
            
            if cfg.env_id == "GridWorld":
                from pc_udrl.envs.gridworld import GridWorld
                viz_env = GridWorld(size=5, seed=cfg.seed)
            else:
                from pc_udrl.envs.gym_wrappers import make_env
                viz_env = make_env(cfg.env_id, seed=cfg.seed)
        except Exception as e:
            print(f"Failed to setup live viz: {e}")
            render_interval = 0

    for epoch in range(epochs):
        agent.train()
        total_train = 0.0
        n_train = 0
        for s, h, dr, a in train_loader:
            s = s.to(device)
            h = h.to(device)
            dr = dr.to(device)
            logits = agent(s, h, dr)
            if discrete:
                loss = nn.functional.cross_entropy(logits, a.long().to(device))
            else:
                loss = nn.functional.mse_loss(logits, a.float().to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_train += float(loss.item()) * s.size(0)
            n_train += s.size(0)
        train_loss = total_train / max(1, n_train)
        
        agent.eval()
        total_val = 0.0
        n_val = 0
        with torch.no_grad():
            for s, h, dr, a in val_loader:
                s = s.to(device)
                h = h.to(device)
                dr = dr.to(device)
                logits = agent(s, h, dr)
                if discrete:
                    loss = nn.functional.cross_entropy(logits, a.long().to(device))
                else:
                    loss = nn.functional.mse_loss(logits, a.float().to(device))
                total_val += float(loss.item()) * s.size(0)
                n_val += s.size(0)
        val_loss = total_val / max(1, n_val)
        logger.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        
        # Periodic Evaluation
        if (epoch + 1) % cfg.eval_interval == 0:
            print(f"Running evaluation at epoch {epoch+1}...")
            
            # Lightweight Eval
            if cfg.env_id == "GridWorld":
                from pc_udrl.envs.gridworld import GridWorld
                eval_env = GridWorld(size=5, seed=cfg.seed)
            else:
                from pc_udrl.envs.gym_wrappers import make_env
                eval_env = make_env(cfg.env_id, seed=cfg.seed)
            
            total_ret = 0.0
            total_gap = 0.0
            # Determine target return dynamically
            if hasattr(dataset, "data") and "returns" in dataset.data:
                 target_return = float(dataset.data["returns"].max().item())
            else:
                 target_return = 100.0 if cfg.env_id == "GridWorld" else 200.0
            
            for _ in range(cfg.eval_episodes):
                s = eval_env.reset()
                done = False
                steps = 0
                ep_ret = 0.0
                ep_gap = 0.0
                gap_steps = 0
                
                while not done and steps < cfg.max_steps:
                    st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                    hh = torch.tensor([cfg.max_steps - steps], dtype=torch.float32, device=device)
                    
                    # Pessimism Calculation
                    cap_val = target_return
                    if pessimist is not None:
                        with torch.no_grad():
                            cap = pessimist(st)
                            cap_val = float(cap.squeeze().detach().cpu().item())
                            gap = max(0.0, target_return - cap_val)
                            ep_gap += gap
                            gap_steps += 1
                    
                    # Command: min(Target, Cap)
                    cmd_val = min(target_return, cap_val)
                    cmd = torch.tensor([cmd_val], dtype=torch.float32, device=device) 
                    
                    with torch.no_grad():
                        a = agent.act(st, hh, cmd)
                    
                    if discrete:
                        a_act = int(a.squeeze().cpu().numpy())
                        s, r, done, _ = eval_env.step(a_act)
                    else:
                        a_act = a.squeeze().cpu().numpy()
                        s, r, done, _ = eval_env.step(a_act)
                    ep_ret += r
                    steps += 1
                
                total_ret += ep_ret
                if gap_steps > 0:
                    total_gap += (ep_gap / gap_steps)
            
            avg_ret = total_ret / cfg.eval_episodes
            avg_gap = total_gap / cfg.eval_episodes
            
            eval_logger.log({"epoch": epoch + 1, "avg_return": avg_ret, "avg_pessimism_gap": avg_gap})
            print(f"Epoch {epoch+1} Eval Return: {avg_ret:.2f}, Gap: {avg_gap:.2f}")
        
        # Live Visualization (Updated)
        if render_interval > 0 and (epoch + 1) % render_interval == 0 and viz_env is not None:
            print(f"Epoch {epoch+1}: Running Live Visualization...")
            try:
                import matplotlib.pyplot as plt # Keep import here for safety if render_interval was 0 initially
                s = viz_env.reset()
                done = False
                steps = 0
                while not done and steps < cfg.max_steps:
                    st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                    h = torch.tensor([cfg.max_steps - steps], dtype=torch.float32, device=device)
                    cmd = torch.tensor([target_return], dtype=torch.float32, device=device) 
                    
                    with torch.no_grad():
                        a = agent.act(st, h, cmd)
                    a_np = a.squeeze(0).detach().cpu().numpy()
                    
                    if discrete:
                        s, r, done, _ = viz_env.step(int(a_np))
                    else:
                        s, r, done, _ = viz_env.step(a_np)
                        
                    frame = viz_env.render()
                    if frame is not None:
                        if im is None:
                            im = ax.imshow(frame)
                            ax.set_axis_off()
                            ax.set_title(f"Epoch {epoch+1} Live")
                        else:
                            im.set_data(frame)
                            ax.set_title(f"Epoch {epoch+1} Live")
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                        # plt.pause(0.01) # Avoid pause if flush_events works better for persistence
                        plt.pause(0.01)
                    steps += 1
            except Exception as e:
                print(f"Live viz update failed: {e}")

    if fig:
        import matplotlib.pyplot as plt # Ensure plt is imported if fig exists
        plt.close(fig)
    save_checkpoint(agent, os.path.join(log_dir, "agent.pt"))
    return agent


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from config import make_config
    from pc_udrl.data.dataset import OfflineDataset
    from pc_udrl.utils import ensure_dirs

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1)
    args = parser.parse_args()

    cfg = make_config(args.phase)
    ensure_dirs(cfg)
    
    dataset = OfflineDataset(cfg)
    train_udrl(cfg, dataset)
