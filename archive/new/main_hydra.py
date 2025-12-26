import os
import hydra
from omegaconf import DictConfig

from pc_udrl.utils import ensure_dirs
from pc_udrl.utils import set_global_seed
from scripts.train_pessimist import train_pessimist
from scripts.train_udrl import train_udrl
# from training.baselines import train_td3_bc, train_iql, train_cql, train_decision_transformer
from scripts.evaluate import evaluate
from pc_udrl.envs.gridworld import GridWorld
from pc_udrl.envs.gym_wrappers import make_env
from pc_udrl.data.dataset import OfflineDataset


def _complete_cfg(cfg: DictConfig):
    if cfg.get("phase") == 1:
        cfg.env_id = "GridWorld"
        if not cfg.get("method"):
            cfg.method = "quantile"
    elif cfg.get("phase") == 2:
        cfg.env_id = "LunarLanderContinuous-v2"
        if not cfg.get("method"):
            cfg.method = "quantile"
    elif cfg.get("phase") == 3:
        cfg.env_id = "LunarLanderContinuous-v2"
        if not cfg.get("method"):
            cfg.method = "cvae"
    if not cfg.get("mode"):
        cfg.mode = "train"


def make_active_env(cfg: DictConfig):
    if cfg.env_id == "GridWorld":
        return GridWorld(size=5, seed=cfg.seed)
    return make_env(cfg.env_id, cfg.seed)


def run_generate(cfg: DictConfig):
    ensure_dirs(cfg)
    if cfg.env_id == "GridWorld":
        from scripts.generate_dataset import main as gen
        gen(cfg.phase)
    else:
        from scripts.generate_baseline import main as genb
        genb(cfg.phase)


def run_train(cfg: DictConfig):
    ensure_dirs(cfg)
    env = make_active_env(cfg)
    dataset = OfflineDataset(cfg)
    if not dataset.exists():
        dataset.generate(env)
    pessimist = train_pessimist(cfg, dataset)
    agent = train_udrl(cfg, dataset)
    return agent, pessimist, env


def run_eval(cfg: DictConfig):
    ensure_dirs(cfg)
    evaluate(cfg, capture_video=getattr(cfg, "capture_video", False), show=getattr(cfg, "show", False))


def run_baselines(cfg: DictConfig):
    ensure_dirs(cfg)
    env = make_active_env(cfg)
    dataset = OfflineDataset(cfg)
    if not dataset.exists():
        dataset.generate(env)
    algs = getattr(cfg, "baselines", ["td3bc", "iql", "cql", "dt"]) 
    # if "td3bc" in algs:
    #     train_td3_bc(cfg, dataset)
    # if "iql" in algs:
    #     train_iql(cfg, dataset)
    # if "cql" in algs:
    #     train_cql(cfg, dataset)
    # if "dt" in algs:
    #     train_decision_transformer(cfg, dataset)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    _complete_cfg(cfg)
    set_global_seed(cfg.seed)
    if cfg.mode == "generate":
        run_generate(cfg)
    elif cfg.mode == "train":
        run_train(cfg)
    elif cfg.mode == "eval":
        run_eval(cfg)
    elif cfg.mode == "baselines":
        run_baselines(cfg)


if __name__ == "__main__":
    main()
