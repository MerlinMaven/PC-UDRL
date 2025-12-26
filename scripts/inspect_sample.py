import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import make_config
from pc_udrl.data.dataset import OfflineDataset

def inspect_sample(phase: int):
    cfg = make_config(phase)
    dataset = OfflineDataset(cfg)
    
    if not dataset.exists():
        print(f"No dataset found for Phase {phase}")
        return

    print("Loading dataset...")
    data = dataset.load()
    obs = data["obs"].numpy()
    
    with open("sample_data.md", "w") as f:
        f.write("# Data Sample\n")
        f.write(f"| {'Idx':<6} | {'Horizon':<8} | {'Return (TG)':<12} | {'State':<15} | {'Action':<6} | {'Reward':<6} |\n")
        f.write(f"|{'-'*8}|{'-'*10}|{'-'*14}|{'-'*17}|{'-'*8}|{'-'*8}|\n")
        
        # 10 Samples from indices 0 to len-1
        indices = np.random.choice(len(obs), 10, replace=False)
        for idx in indices:
            s_str = f"[{obs[idx][0]:.0f}, {obs[idx][1]:.0f}]"
            rtg = data['returns'][idx]
            h = data['horizons'][idx]
            a = data['actions'][idx]
            r = data['rewards'][idx]
            f.write(f"| {idx:<6} | {h:<8.0f} | {rtg:<12.1f} | {s_str:<15} | {a:<6} | {r:<6.1f} |\n")
    print("Written to sample_data.md")

if __name__ == "__main__":
    inspect_sample(1)
