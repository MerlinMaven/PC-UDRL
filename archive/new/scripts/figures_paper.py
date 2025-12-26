import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main(phase: int):
    base = os.path.join("runs", f"phase{phase}")
    out = os.path.join(base, "paper_figs")
    os.makedirs(out, exist_ok=True)
    files = {
        "agent": os.path.join(base, "agent.csv"),
        "pessimist": os.path.join(base, "pessimist.csv"),
        "eval": os.path.join(base, "eval.csv"),
        "td3bc": os.path.join(base, "td3bc.csv"),
        "iql": os.path.join(base, "iql.csv"),
        "cql": os.path.join(base, "cql.csv"),
        "dt": os.path.join(base, "dt.csv"),
    }
    dfs = {k: (pd.read_csv(v) if os.path.exists(v) else None) for k, v in files.items()}
    sns.set_theme(style="whitegrid")
    if dfs["agent"] is not None and dfs["pessimist"] is not None:
        plt.figure(figsize=(6, 4), dpi=300)
        plt.plot(dfs["agent"]["epoch"], dfs["agent"]["train_loss"], label="Agent Train")
        plt.plot(dfs["agent"]["epoch"], dfs["agent"]["val_loss"], label="Agent Val")
        plt.plot(dfs["pessimist"]["epoch"], dfs["pessimist"]["train_loss"], label="Pessimist Train")
        plt.plot(dfs["pessimist"]["epoch"], dfs["pessimist"]["val_loss"], label="Pessimist Val")
        plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.tight_layout()
        plt.savefig(os.path.join(out, "fig1_losses.png")); plt.close()
    if dfs["eval"] is not None:
        plt.figure(figsize=(6, 4), dpi=300)
        plt.plot([1], [dfs["eval"]["avg_return"].iloc[-1]], marker="o", label="Mean Return")
        plt.legend(); plt.xlabel("Run"); plt.ylabel("Return"); plt.tight_layout()
        plt.savefig(os.path.join(out, "fig2_eval_return.png")); plt.close()
        plt.figure(figsize=(6, 4), dpi=300)
        plt.plot([1], [dfs["eval"]["avg_pessimism_gap"].iloc[-1]], marker="o", label="Pessimism Gap")
        plt.legend(); plt.xlabel("Run"); plt.ylabel("Gap"); plt.tight_layout()
        plt.savefig(os.path.join(out, "fig3_gap.png")); plt.close()
    # Baselines figures
    for key in ["td3bc", "iql", "cql", "dt"]:
        if dfs[key] is not None:
            plt.figure(figsize=(6, 4), dpi=300)
            y = [c for c in dfs[key].columns if c not in ("epoch", )]
            for col in y:
                plt.plot(dfs[key]["epoch"], dfs[key][col], label=col)
            plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Metric"); plt.tight_layout()
            plt.savefig(os.path.join(out, f"fig_{key}.png")); plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1)
    args = parser.parse_args()
    main(args.phase)
