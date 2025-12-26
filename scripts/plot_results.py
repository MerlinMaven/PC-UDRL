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


def main(phase: int, runs_dir: str = "runs"):
    base = os.path.join(runs_dir, f"phase{phase}")
    agent_csv = os.path.join(base, "agent.csv")
    agent_csv_alt = os.path.join(base, "agent_log.csv")
    pess_csv = os.path.join(base, "pessimist.csv")
    pess_csv_alt = os.path.join(base, "pessimist_log.csv")
    eval_csv = os.path.join(base, "eval.csv")
    eval_csv_alt = os.path.join(base, "eval_pessimist.csv")
    # Updated structure: Save plots to 'outputs/plots'
    # We can infer project root from runs_dir
    project_root = os.path.dirname(os.path.abspath(runs_dir))
    out_dir = os.path.join(project_root, "outputs", "plots", f"phase{phase}")
    os.makedirs(out_dir, exist_ok=True)

    agent_df = _read_any(agent_csv, agent_csv_alt)
    pess_df = _read_any(pess_csv, pess_csv_alt)
    eval_df = _read_any(eval_csv, eval_csv_alt)

    sns.set_theme(style="whitegrid")

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

    if eval_df is not None and "avg_return" in eval_df.columns:
        plt.figure(figsize=(8, 5), dpi=200)
        plt.plot(eval_df.index + 1, eval_df["avg_return"], label="Eval Mean Return")
        plt.xlabel("Run Index")
        plt.ylabel("Mean Return")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "eval_mean_return.png"))
        plt.close()

    if eval_df is not None and "avg_pessimism_gap" in eval_df.columns:
        plt.figure(figsize=(8, 5), dpi=200)
        plt.plot(eval_df.index + 1, eval_df["avg_pessimism_gap"], label="Pessimism Gap")
        plt.xlabel("Run Index")
        plt.ylabel("Gap")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "pessimism_gap.png"))
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1)
    args = parser.parse_args()
    main(args.phase)
