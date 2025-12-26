from typing import Dict, Any

import numpy as np


def normalized_score(env_name: str, returns: np.ndarray) -> float:
    try:
        import d4rl
        import gymnasium as gym
        env = gym.make(env_name)
        return float(env.get_normalized_score(returns.mean()))
    except Exception:
        return float(returns.mean())


def cvar(returns: np.ndarray, alpha: float = 0.1) -> float:
    k = max(1, int(alpha * len(returns)))
    vals = np.sort(returns)[:k]
    return float(vals.mean())


def worst_case(returns: np.ndarray) -> float:
    return float(np.min(returns))


def ood_detection_accuracy(scores_in: np.ndarray, scores_out: np.ndarray, threshold: float) -> float:
    tp = (scores_out < threshold).sum()
    tn = (scores_in >= threshold).sum()
    acc = (tp + tn) / (len(scores_in) + len(scores_out))
    return float(acc)

