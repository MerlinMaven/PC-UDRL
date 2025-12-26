import os
import argparse
import numpy as np

from config import make_config
from pc_udrl.d4rl.wrapper import load_d4rl
from scripts.train_pessimist import train_pessimist
from scripts.train_udrl import train_udrl
from pc_udrl.metrics.metrics import normalized_score, cvar, worst_case


def run(envs, method: str, epochs: int):
    results = []
    for env_id in envs:
        cfg = make_config(2)
        cfg.env_id = env_id
        cfg.method = method
        cfg.epochs = epochs
        data, info = load_d4rl(env_id, normalize=True)
        from pc_udrl.data.dataset import OfflineDataset
        ds = OfflineDataset(cfg)
        ds.data = data
        pessimist = train_pessimist(cfg, ds)
        agent = train_udrl(cfg, ds)
        returns = ds.data["returns"].detach().cpu().numpy()
        ns = normalized_score(env_id, returns)
        cv = cvar(returns, alpha=0.1)
        wc = worst_case(returns)
        results.append((env_id, ns, cv, wc))
        print(env_id, ns, cv, wc)
    out_dir = os.path.join("runs", f"phase{2}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "d4rl_benchmark.csv"), "w") as f:
        f.write("env,normalized,cvar,worst\n")
        for env, ns, cv, wc in results:
            f.write(f"{env},{ns},{cv},{wc}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="quantile")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    envs = [
        "hopper-medium-v2",
        "walker2d-medium-v2",
        "halfcheetah-medium-v2",
    ]
    run(envs, args.method, args.epochs)
