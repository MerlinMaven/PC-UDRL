
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import make_config
from pc_udrl.utils import get_device
from pc_udrl.pessimists.quantile import QuantileRegressor
from pc_udrl.agents.udrl_agent import UDRLAgent

def visualize_expert(cfg):
    print("Generating Expert Visualizations for GridWorld...")
    device = get_device(cfg)
    
    # Load Models
    log_dir = os.path.join(cfg.runs_dir, f"phase{cfg.phase}")
    
    # Pessimist
    state_dim = 2
    pessimist = QuantileRegressor(state_dim=state_dim, hidden_dim=cfg.hidden_dim, q=cfg.pessimist_quantile, cfg=cfg).to(device)
    pess_path = os.path.join(log_dir, "pessimist.pt")
    if os.path.exists(pess_path):
        pessimist.load_state_dict(torch.load(pess_path, map_location=device))
        print(f"Loaded Pessimist from {pess_path}")
    else:
        print(f"Warning: Pessimist checkpoint not found at {pess_path}")
        return

    # Agent
    # For GridWorld, action_dim=4, discrete=True
    agent = UDRLAgent(state_dim=state_dim, action_dim=4, discrete=True, hidden_dim=cfg.hidden_dim).to(device)
    agent_path = os.path.join(log_dir, "agent.pt")
    if os.path.exists(agent_path):
        agent.load_state_dict(torch.load(agent_path, map_location=device))
        print(f"Loaded Agent from {agent_path}")
    else:
        print(f"Warning: Agent checkpoint not found at {agent_path}")
        return

    pessimist.eval()
    agent.eval()
    
    # Prepare Grid
    size = 5
    grid_coords = []
    for y in range(size):
        for x in range(size):
            grid_coords.append([y, x])
    grid_coords_np = np.array(grid_coords)
    tensor_coords = torch.tensor(grid_coords_np, dtype=torch.float32).to(device)

    # Output Dir
    plot_dir = os.path.join(log_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Pessimist Heatmap
    print("Generating Pessimist Heatmap...")
    with torch.no_grad():
        values = pessimist(tensor_coords).cpu().numpy()
    
    value_grid = values.reshape((size, size))
    
    plt.figure(figsize=(6, 5), dpi=150)
    sns.heatmap(value_grid, annot=True, fmt=".1f", cmap="viridis", cbar_kws={'label': 'Predicted Return (Cap)'})
    plt.title("Pessimistic Return Cap (Quantile)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(os.path.join(plot_dir, "heatmap_pessimist.png"))
    plt.close()

    # 2. Policy Vector Field
    print("Generating Policy Vector Field...")
    
    u_vec = np.zeros((size, size))
    v_vec = np.zeros((size, size))
    
    with torch.no_grad():
        # Get caps again to feed as command
        caps = pessimist(tensor_coords) # (25,)
        
        # Horizon command: Fixed horizon of 10 for visualization
        horizons = torch.full((len(grid_coords),), 10.0, device=device)
        
        # Agent forward
        logits = agent(tensor_coords, horizons, caps) # (25, 4)
        probs = torch.softmax(logits, dim=-1)
        actions = torch.argmax(probs, dim=-1).cpu().numpy() # (25,)
    
    for idx, (y, x) in enumerate(grid_coords):
        a = actions[idx]
        if a == 0: # Up (y-1) -> dy = -1
            u, v = 0, -1 
        elif a == 1: # Down (y+1) -> dy = 1
            u, v = 0, 1 
        elif a == 2: # Left (x-1) -> dx = -1
            u, v = -1, 0
        elif a == 3: # Right (x+1) -> dx = 1
            u, v = 1, 0
        
        u_vec[y, x] = u
        v_vec[y, x] = v

    plt.figure(figsize=(6, 6), dpi=150)
    # We plot the grid with Y increasing downwards (default matrix view), 
    # so we use invert_yaxis() to match the heatmap/gridworld coords.
    # Quiver X, Y are coordinates.
    plt.quiver(
        [c[1] for c in grid_coords],  # X coordinates
        [c[0] for c in grid_coords],  # Y coordinates
        u_vec.flatten(), 
        v_vec.flatten(),
        pivot='mid', scale=15, width=0.006, color='black'
    )
    # Configure grid visuals
    plt.xlim(-0.5, size - 0.5)
    plt.ylim(size - 0.5, -0.5) # Invert Y limits manually or use invert_yaxis
    # plt.gca().invert_yaxis() # Already handled by ylim ordering above possibly, but let's be explicit
    
    # Overlay goal and start roughly
    plt.text(4, 4, 'GOAL', color='green', ha='center', va='center', fontweight='bold')
    plt.text(0, 0, 'START', color='blue', ha='center', va='center', fontweight='bold')
    
    plt.title("Policy Vector Field\n(Command: Horizon=10, Return=PessimistCap)")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "policy_vector_field.png"))
    plt.close()

    print("Expert visualizations saved.")

if __name__ == "__main__":
    cfg = make_config(1)
    visualize_expert(cfg)
