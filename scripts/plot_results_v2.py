import os
import sys
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import glob

def _read_any(path1, path2):
    if os.path.exists(path1):
        return pd.read_csv(path1)
    if os.path.exists(path2):
        return pd.read_csv(path2)
    return None

def _read_d3rlpy_baseline(algo_name, phase_dir):
    # d3rlpy logs are in d3rlpy_logs/runs/phase{phase}/baselines/{algo}/environment.csv
    # We need to construct the path relative to project root
    # phase_dir is usually runs/phase2
    # So we need to go up to project root
    project_root = os.path.dirname(os.path.dirname(phase_dir))
    
    # Check standard d3rlpy locations
    # 1. d3rlpy_logs/runs/phaseX/baselines/ALGO/environment.csv
    base_log = os.path.join(project_root, "d3rlpy_logs", "runs", f"phase{phase}", "baselines", algo_name)
    env_csv = os.path.join(base_log, "environment.csv")
    
    if os.path.exists(env_csv):
        try:
            # d3rlpy csv: epoch, step, reward
            df = pd.read_csv(env_csv, header=None)
            # Columns are typically: index, step, return
            # Let's assume col 0 is index/epoch, col 2 is return
            df.columns = ["epoch", "step", "return"]
            return df
        except Exception as e:
            print(f"Failed to read baseline {algo_name}: {e}")
            return None
    return None

def main(phase: int, runs_dir: str = "runs"):
    base = os.path.join(runs_dir, f"phase{phase}")
    agent_csv = os.path.join(base, "agent.csv")
    agent_csv_alt = os.path.join(base, "agent_log.csv")
    pess_csv = os.path.join(base, "pessimist.csv")
    pess_csv_alt = os.path.join(base, "pessimist_log.csv")
    eval_csv = os.path.join(base, "eval.csv")
    eval_csv_alt = os.path.join(base, "eval_pessimist.csv")
    
    # Updated structure: Save plots to 'outputs/plots'
    project_root = os.path.dirname(os.path.abspath(runs_dir))
    out_dir = os.path.join(project_root, "outputs", "plots", f"phase{phase}")
    os.makedirs(out_dir, exist_ok=True)

    agent_df = _read_any(agent_csv, agent_csv_alt)
    pess_df = _read_any(pess_csv, pess_csv_alt)
    eval_df = _read_any(eval_csv, eval_csv_alt)

    sns.set_theme(style="whitegrid")

    # Plot Losses (Unchanged)
    if agent_df is not None:
        plt.figure(figsize=(8, 5), dpi=200)
        plt.plot(agent_df["epoch"], agent_df["train_loss"], label="Agent Train Loss")
        plt.plot(agent_df["epoch"], agent_df["val_loss"], label="Agent Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "agent_loss.png"))
        plt.close()

    if pess_df is not None:
        plt.figure(figsize=(8, 5), dpi=200)
        plt.plot(pess_df["epoch"], pess_df["train_loss"], label="Pessimist Train Loss")
        plt.plot(pess_df["epoch"], pess_df["val_loss"], label="Pessimist Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "pessimist_loss.png"))
        plt.close()

    # Comparable Evaluation Plot (PC-UDRL vs Baselines)
    plt.figure(figsize=(10, 6), dpi=200)
    
    # Plot PC-UDRL
    if eval_df is not None and "avg_return" in eval_df.columns:
        plt.plot(eval_df["epoch"], eval_df["avg_return"], label="PC-UDRL (Ours)", linewidth=2.5, marker='o')
    
    # Plot Baselines
    baselines = ["cql", "iql", "td3plusbc"]
    colors = {"cql": "red", "iql": "green", "td3plusbc": "orange"}
    
    global phase # Needed for helper? No, passing phase in main args to helper would be cleaner but I hardcoded inside helper for now
    # Wait, helper _read_d3rlpy_baseline needs phase. I'll modify it to use the outer 'phase' variable or pass it.
    
    pass

# Redefine helper correctly outside or pass phase
def _read_d3rlpy_baseline_v2(algo_name, project_root, phase):
    base_log = os.path.join(project_root, "d3rlpy_logs", "runs", f"phase{phase}", "baselines", algo_name)
    env_csv = os.path.join(base_log, "environment.csv")
    if os.path.exists(env_csv):
        try:
            df = pd.read_csv(env_csv, header=None)
            df.columns = ["epoch", "step", "return"]
            return df
        except:
            return None
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--runs_dir", type=str, default="runs")
    args = parser.parse_args()
    
    phase = args.phase
    runs_dir = args.runs_dir
    
    # ... logic repeated ...
    # Rewriting full clean script in one block below
