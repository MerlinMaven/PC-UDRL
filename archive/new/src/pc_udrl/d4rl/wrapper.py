from typing import Any, Dict, Tuple

import numpy as np
import torch


class Normalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + self.eps)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * (self.std + self.eps) + self.mean


def _maybe_import_d4rl():
    try:
        import d4rl
        return d4rl
    except Exception:
        return None


def load_d4rl(env_name: str, normalize: bool = True) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    d4rl = _maybe_import_d4rl()
    if d4rl is None:
        raise RuntimeError("d4rl is not installed; please install d4rl and mujoco dependencies")
    import gymnasium as gym
    env = gym.make(env_name)
    dataset = env.get_dataset()
    obs = dataset["observations"].astype(np.float32)
    actions = dataset["actions"].astype(np.float32)
    rewards = dataset["rewards"].astype(np.float32)
    terminals = dataset["terminals"].astype(np.bool_)
    horizons = np.ones_like(rewards)
    returns = np.zeros_like(rewards, dtype=np.float32)
    gamma = 0.99
    R = 0.0
    for i in reversed(range(len(rewards))):
        R = rewards[i] + gamma * R * (1.0 - float(terminals[i]))
        returns[i] = R
        horizons[i] = 1 if terminals[i] else horizons[i] + 1
    info: Dict[str, Any] = {}
    if normalize:
        s_mean = obs.mean(axis=0)
        s_std = obs.std(axis=0)
        a_mean = actions.mean(axis=0)
        a_std = actions.std(axis=0)
        s_norm = Normalizer(s_mean, s_std)
        a_norm = Normalizer(a_mean, a_std)
        obs = s_norm.normalize(obs)
        actions = a_norm.normalize(actions)
        info["state_normalizer"] = s_norm
        info["action_normalizer"] = a_norm
    data = {
        "obs": torch.tensor(obs, dtype=torch.float32),
        "actions": torch.tensor(actions, dtype=torch.float32),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "returns": torch.tensor(returns, dtype=torch.float32),
        "horizons": torch.tensor(horizons, dtype=torch.float32),
    }
    return data, info

