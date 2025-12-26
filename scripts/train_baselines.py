import os
import sys
import argparse
import d3rlpy
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import make_config

def train_baseline(cfg, algo_name, epochs=30):
    dataset_path = os.path.join(cfg.dataset_dir, "phase2", "dataset.h5")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Load Dataset (d3rlpy can read H5 if formatted correctly, or we load manually)
    # Since our H5 is custom, we load manually and converting to MDPDataset
    import h5py
    with h5py.File(dataset_path, "r") as f:
        observations = f["observations"][:]
        actions = f["actions"][:]
        rewards = f["rewards"][:]
        terminals = f["terminals"][:]
        # Timeouts?
        if "timeouts" in f:
            timeouts = f["timeouts"][:]
        else:
            timeouts = np.zeros_like(terminals) # Assume none

    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
    )
    
    # Fix rewards if needed (some offline algorithms prefer normalized or small rewards)
    # But we keep raw for comparison fairness
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training {algo_name.upper()} on {device}...")
    
    if algo_name == "cql":
        algo = d3rlpy.algos.CQLConfig().create(device=device)
    elif algo_name == "iql":
        algo = d3rlpy.algos.IQLConfig(
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            weight_temp=3.0,  # Common for Ant/Walker, might need tuning
        ).create(device=device)
    elif algo_name == "td3plusbc":
        algo = d3rlpy.algos.TD3PlusBCConfig(
            alpha=2.5,
        ).create(device=device)
    else:
        raise ValueError(f"Unknown algo: {algo_name}")

    # Set logging
    log_dir = os.path.join(cfg.runs_dir, f"phase{cfg.phase}", "baselines", algo_name)
    
    # Train
    # 1 epoch = 'n_steps_per_epoch' updates. 
    # Total steps = epochs * n_steps_per_epoch
    # We want roughly same compute as our agent? 
    # Let's do 30 epochs * 1000 steps = 30k updates.
    n_steps = epochs * 1000
    
    algo.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=1000,
        save_interval=10000, # Save every 10 epochs
        evaluators={
            "environment": d3rlpy.metrics.EnvironmentEvaluator(gym.make(cfg.env_id))
        },
        experiment_name=log_dir, # d3rlpy uses this as logdir base
        with_timestamp=False,
    )
    
    # Save final model
    model_path = os.path.join(log_dir, "model.d3")
    algo.save_model(model_path)
    print(f"Saved {algo_name} model to {model_path}")

if __name__ == "__main__":
    import gymnasium as gym # Imported inside scorer but good to have here
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, choices=["cql", "iql", "td3plusbc"])
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    
    cfg = make_config(2)
    train_baseline(cfg, args.algo, args.epochs)
