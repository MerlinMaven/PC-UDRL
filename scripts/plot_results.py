import os
import sys
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def _read_any(path1, path2):
    if os.path.exists(path1):
        return pd.read_csv(path1)
    if os.path.exists(path2):
        return pd.read_csv(path2)
    return None

def _read_d3rlpy_baseline(algo_name, project_root, phase):
    # Try to find d3rlpy logs
    # Location: d3rlpy_logs/runs/phase{phase}/baselines/{algo}/environment.csv
    base_log = os.path.join(project_root, "d3rlpy_logs", "runs", f"phase{phase}", "baselines", algo_name)
    env_csv = os.path.join(base_log, "environment.csv")
    
    if os.path.exists(env_csv):
        try:
            # d3rlpy csv has no header usually: epoch, step, reward
            df = pd.read_csv(env_csv, header=None)
            # Check shape or columns
            if df.shape[1] >= 3:
                df = df.iloc[:, [0, 2]] # Epoch, Return
                df.columns = ["epoch", "return"]
                return df
        except Exception as e:
            print(f"Error reading {algo_name}: {e}")
    return None

def main(phase: int, runs_dir: str = "runs"):
    base = os.path.join(runs_dir, f"phase{phase}")
    agent_df = _read_any(os.path.join(base, "agent.csv"), os.path.join(base, "agent_log.csv"))
    pess_df = _read_any(os.path.join(base, "pessimist.csv"), os.path.join(base, "pessimist_log.csv"))
    eval_df = _read_any(os.path.join(base, "eval.csv"), os.path.join(base, "eval_pessimist.csv"))

    project_root = os.path.dirname(os.path.abspath(runs_dir))
    out_dir = os.path.join(project_root, "outputs", "plots", f"phase{phase}")
    os.makedirs(out_dir, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # 1. Agent Loss
    if agent_df is not None:
        plt.figure(figsize=(8, 5), dpi=200)
        plt.plot(agent_df["epoch"], agent_df["train_loss"], label="Train Loss")
        plt.plot(agent_df["epoch"], agent_df["val_loss"], label="Val Loss")
        plt.title("UDRL Agent Loss")
        plt.legend()
        plt.savefig(os.path.join(out_dir, "agent_loss.png"))
        plt.close()

    # 2. Pessimist Loss
    if pess_df is not None:
        plt.figure(figsize=(8, 5), dpi=200)
        plt.plot(pess_df["epoch"], pess_df["train_loss"], label="Train Loss")
        plt.plot(pess_df["epoch"], pess_df["val_loss"], label="Val Loss")
        plt.title("Pessimist Loss")
        plt.legend()
        plt.savefig(os.path.join(out_dir, "pessimist_loss.png"))
        plt.close()

    # 3. Comparative Evaluation (The Big One)
    plt.figure(figsize=(10, 6), dpi=200)
    
    # Baselines
    baselines = {
        "cql": ("Conservative Q-Learning (CQL)", "red"),
        "iql": ("Implicit Q-Learning (IQL)", "green"),
        "td3plusbc": ("TD3+BC", "orange")
    }
    
    for algo, (label, color) in baselines.items():
        df = _read_d3rlpy_baseline(algo, project_root, phase)
        if df is not None:
            plt.plot(df["epoch"], df["return"], label=label, color=color, linestyle="--", alpha=0.7)
            
    # Our Method
    if eval_df is not None and "avg_return" in eval_df.columns:
        # Assuming eval_df has 'epoch' column, if not create from index
        epochs = eval_df["epoch"] if "epoch" in eval_df.columns else range(1, len(eval_df)+1)
        plt.plot(epochs, eval_df["avg_return"], label="PC-UDRL (Ours)", color="blue", linewidth=3)

    plt.title(f"Phase {phase}: Offline RL Benchmark (LunarLander)")
    plt.xlabel("Epoch")
    plt.ylabel("Average Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "benchmark_comparison.png"))
    plt.close()
    
    print(f"Plots saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--runs_dir", type=str, default="runs")
    args = parser.parse_args()
    main(args.phase, args.runs_dir)
