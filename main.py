import os
import argparse
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from config import make_config
from pc_udrl.utils import ensure_dirs
from scripts.train_pessimist import train_pessimist
from scripts.train_udrl import train_udrl
from scripts.evaluate import evaluate
from pc_udrl.envs.gridworld import GridWorld
from pc_udrl.envs.gym_wrappers import make_env
from pc_udrl.data.dataset import OfflineDataset
from pc_udrl.utils import set_global_seed
from scripts.generate_dataset import main as generate_dataset
# from scripts.train_baseline import train_baseline


def make_active_env(cfg):
    if cfg.env_id == "GridWorld":
        return GridWorld(size=5, seed=cfg.seed)
    return make_env(cfg.env_id, cfg.seed)


def run_generate(cfg):
    ensure_dirs(cfg)
    # Always use the unified generation script
    generate_dataset(cfg.phase)


def run_train(cfg, algo=None):
    ensure_dirs(cfg)
    # Check if baseline requested
    if algo in ["cql", "iql", "td3plusbc"]:
        train_baseline(cfg, algo)
        return

    env = make_active_env(cfg)
    dataset = OfflineDataset(cfg)
    if not dataset.exists():
        dataset.generate(env)
    pessimist = train_pessimist(cfg, dataset)
    # Pass config as is; train_udrl uses attributes
    agent = train_udrl(cfg, dataset)
    return agent, pessimist, env


def run_eval(cfg):
    ensure_dirs(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline UDRL with Pessimistic Command Generation")
    parser.add_argument("--phase", type=int, default=1, help="Phase 1=GridWorld+Quantile, 2=LunarLanderCont+Quantile, 3=LunarLanderCont+CVAE")
    parser.add_argument("--mode", type=str, choices=["generate", "train", "eval"], default="train", help="Mode: generate dataset, train models, or evaluate")
    parser.add_argument("--capture_video", action="store_true", help="Capture evaluation episodes to video")
    parser.add_argument("--show", action="store_true", help="Display environment during evaluation")
    parser.add_argument("--render", action="store_true", help="Display environment during training (Interactive Mode)")
    parser.add_argument("--method", type=str, choices=["quantile", "cvae", "diffusion"], help="Override pessimistic method")
    parser.add_argument("--algo", type=str, choices=["cql", "iql", "td3plusbc"], help="Baseline algorithm to train")
    args = parser.parse_args()

    cfg = make_config(args.phase)
    if args.method:
        cfg.method = args.method
    if args.render:
        cfg.render = True
        cfg.render_interval = 1 # Render every epoch
    set_global_seed(cfg.seed)
    if args.mode == "generate":
        run_generate(cfg)
    elif args.mode == "train":
        run_train(cfg, algo=args.algo)
    elif args.mode == "eval":
        evaluate(cfg, capture_video=args.capture_video, show=args.show)
