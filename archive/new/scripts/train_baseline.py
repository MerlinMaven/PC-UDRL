import os
import argparse
import d3rlpy
from pc_udrl.data.dataset import OfflineDataset
from config import make_config
from pc_udrl.utils import ensure_dirs, set_global_seed, get_device


def train_baseline(cfg, algo_name: str):
    set_global_seed(cfg.seed)
    ensure_dirs(cfg)
    device = "cuda" if get_device(cfg).type == "cuda" else "cpu"
    
    # Load Dataset
    print(f"Loading dataset for {cfg.env_id}...")
    ds = OfflineDataset(cfg)
    if not ds.exists():
        print("Dataset not found. Please generate it first.")
        return
    data = ds.load()
    
    # Convert to d3rlpy MDPDataset
    observations = data["obs"].numpy()
    actions = data["actions"].numpy()
    rewards = data["rewards"].numpy()
    terminals = data["dones"].numpy()
    
    # d3rlpy expects episodes, not flat buffers usually, but supports MDPDataset
    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )
    
    # Select Algorithm
    print(f"Initializing {algo_name}...")
    if algo_name == "cql":
        algo = d3rlpy.algos.CQLConfig(
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            batch_size=256,
        ).create(device=device)
    elif algo_name == "iql":
        algo = d3rlpy.algos.IQLConfig(
            batch_size=256,
        ).create(device=device)
    elif algo_name == "td3plusbc":
        algo = d3rlpy.algos.TD3PlusBCConfig(
            batch_size=256,
        ).create(device=device)
    else:
        raise ValueError(f"Unknown baseline: {algo_name}")
    
    # Train
    print(f"Starting training for {cfg.epochs} epochs...")
    # d3rlpy uses n_steps or n_epochs. We'll map cfg.epochs to steps approx
    # Use len(observations) as dataset size
    steps_per_epoch = len(observations) // 256
    n_steps = cfg.epochs * steps_per_epoch
    
    log_dir = os.path.join(cfg.runs_dir, f"phase{cfg.phase}", "baselines", algo_name)
    os.makedirs(log_dir, exist_ok=True)
    
    algo.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=steps_per_epoch,
        save_interval=steps_per_epoch,
        experiment_name=f"phase{cfg.phase}_{algo_name}",
        with_timestamp=False,
        # verbose=True removed for d3rlpy 2.x compatibility
        logger_adapter=d3rlpy.logging.FileAdapterFactory(root_dir=log_dir)
    )
    
    # Save manually to be sure
    save_path = os.path.join(log_dir, "model.d3")
    algo.save_model(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=2)
    parser.add_argument("--algo", type=str, required=True, choices=["cql", "iql", "td3plusbc"])
    args = parser.parse_args()
    
    cfg = make_config(args.phase)
    train_baseline(cfg, args.algo)
