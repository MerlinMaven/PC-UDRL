import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def main():
    runs_dir = "runs"
    out_dir = "outputs/plots/phase3"
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Load Phase 2 (Quantile) Eval
    p2_path = os.path.join(runs_dir, "phase2", "eval.csv")
    df2 = pd.read_csv(p2_path) if os.path.exists(p2_path) else None
    
    # Load Phase 3 (CVAE) Eval
    p3_path = os.path.join(runs_dir, "phase3", "eval.csv")
    df3 = pd.read_csv(p3_path) if os.path.exists(p3_path) else None
    
    # Load Phase 3 Training Log
    p3_train_path = os.path.join(runs_dir, "phase3", "pessimist.csv")
    df_train = pd.read_csv(p3_train_path) if os.path.exists(p3_train_path) else None

    # 1. CVAE Loss Plot
    if df_train is not None:
        plt.figure(figsize=(8, 5), dpi=200)
        plt.plot(df_train["epoch"], df_train["train_loss"], label="Train ELBO")
        plt.plot(df_train["epoch"], df_train["val_loss"], label="Val ELBO")
        plt.title("Phase 3: CVAE Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(out_dir, "cvae_loss.png"))
        plt.close()

    # 2. Comparative Return Plot
    plt.figure(figsize=(10, 6), dpi=200)
    if df2 is not None:
        plt.plot(df2["epoch"], df2["avg_return"], label="Phase 2 (Quantile)", color="blue", linestyle="--")
    if df3 is not None:
        plt.plot(df3["epoch"], df3["avg_return"], label="Phase 3 (CVAE)", color="purple", linewidth=2.5)
        
    plt.title("PC-UDRL Safety Method Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Average Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phase2_vs_phase3.png"))
    plt.close()
    
    # 3. Gap Comparison
    plt.figure(figsize=(10, 6), dpi=200)
    if df2 is not None and "avg_pessimism_gap" in df2.columns:
        plt.plot(df2["epoch"], df2["avg_pessimism_gap"], label="Phase 2 Gap (Quantile)", color="blue", linestyle="--")
    if df3 is not None and "avg_pessimism_gap" in df3.columns:
        plt.plot(df3["epoch"], df3["avg_pessimism_gap"], label="Phase 3 Gap (CVAE)", color="purple", linewidth=2.5)
        
    plt.title("Pessimism Gap Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Gap (Target - Cap)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gap_comparison.png"))
    plt.close()
    
    print(f"Plots saved to {out_dir}")

if __name__ == "__main__":
    main()
