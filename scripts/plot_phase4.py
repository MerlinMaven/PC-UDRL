import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add src to path if needed (though we just read CSVs here)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import make_config

def main():
    # Setup paths
    cfg = make_config(4) # Phase 4 config has access to runs dir
    runs_dir = cfg.runs_dir
    
    # Load Phase 2 (uDRL + Quantile)
    p2_path = os.path.join(runs_dir, "phase2", "eval.csv")
    if os.path.exists(p2_path):
        df2 = pd.read_csv(p2_path)
    else:
        print(f"Warning: Phase 2 eval not found at {p2_path}")
        df2 = None
        
    # Load Phase 3 (uDRL + CVAE)
    p3_path = os.path.join(runs_dir, "phase3", "eval.csv")
    if os.path.exists(p3_path):
        df3 = pd.read_csv(p3_path)
    else:
        print(f"Warning: Phase 3 eval not found at {p3_path}")
        df3 = None
        
    # Load Phase 4 (uDRL + Diffusion)
    # Note: Phase 4 might not have eval.csv yet if we only ran training. 
    # Logic: We might need to run evaluate.py for Phase 4 first or use training logs if available?
    # Actually, the user flow implies we train then eval. 
    # Let's assume we run evaluate.py first OR we plot what we have.
    # If phase 4 eval is missing, we should probably run an evaluation for it.
    p4_path = os.path.join(runs_dir, "phase4", "eval.csv")
    if os.path.exists(p4_path):
        df4 = pd.read_csv(p4_path)
    else:
        # Create a dummy dataframe if not exists just to show code structure, 
        # but in reality we should run evaluation first.
        print(f"Warning: Phase 4 eval not found at {p4_path}. Using placeholder.")
        df4 = pd.DataFrame({"avg_return": [], "avg_pessimism_gap": []})

    # Prepare Plots
    os.makedirs(os.path.join(cfg.output_dir, "plots", "phase4"), exist_ok=True)
    
    # 1. Comparative Returns
    plt.figure(figsize=(10, 6))
    if df2 is not None:
        plt.plot(df2["avg_return"], label="Phase 2: Quantile (Conservative)", color="blue", linestyle="--")
    if df3 is not None:
        plt.plot(df3["avg_return"], label="Phase 3: CVAE (Optimistic/Failed)", color="red", linestyle="-.")
    if df4 is not None and not df4.empty:
        plt.plot(df4["avg_return"], label="Phase 4: Diffusion (Adaptive)", color="green", linewidth=2)
    
    plt.axhline(y=195, color="black", linestyle=":", label="Target Command (+195)")
    plt.title("Comparative Performance: Quantile vs CVAE vs Diffusion")
    plt.xlabel("Evaluation Epochs")
    plt.ylabel("Average Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(cfg.output_dir, "plots", "phase4", "comparison_returns.png"))
    plt.close()

    # 2. Comparative Pessimism Gaps
    plt.figure(figsize=(10, 6))
    if df2 is not None:
        plt.plot(df2["avg_pessimism_gap"], label="Phase 2 Gap (High)", color="blue", linestyle="--")
    if df3 is not None:
        plt.plot(df3["avg_pessimism_gap"], label="Phase 3 Gap (Zero)", color="red", linestyle="-.")
    if df4 is not None and not df4.empty:
        plt.plot(df4["avg_pessimism_gap"], label="Phase 4 Gap (Adaptive)", color="green", linewidth=2)
        
    plt.title("Pessimism Gap Analysis (Target - SafeCap)")
    plt.xlabel("Evaluation Epochs")
    plt.ylabel("Gap Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(cfg.output_dir, "plots", "phase4", "comparison_gaps.png"))
    plt.close()
    
    print("Plots saved to outputs/plots/phase4")

if __name__ == "__main__":
    main()
