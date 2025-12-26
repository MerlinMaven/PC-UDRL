import numpy as np


def _import_gym():
    try:
        import gymnasium as gym
        return gym
    except ImportError:
        import gym
        return gym


def make_env(env_id: str, seed: int):
    gym = _import_gym()
    try:
        env = gym.make(env_id, render_mode="rgb_array")
    except Exception:
        env = gym.make(env_id)
    try:
        env.reset(seed=seed)
    except Exception:
        pass
    return GymEnvWrapper(env)


class GymEnvWrapper:
    def __init__(self, env):
        self.env = env

    def reset(self):
        out = self.env.reset()
        if isinstance(out, tuple):
            return out[0]
        return out

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            o, r, terminated, truncated, info = out
            done = terminated or truncated
            return o, r, done, info
        return out

    def sample_action(self):
        return self.env.action_space.sample()

    @property
    def observation_space(self):
        shape = self.env.observation_space.shape
        return shape

    @property
    def action_space(self):
        space = self.env.action_space
        if hasattr(space, "n"):
            return space.n
        return space.shape[0]

    @property
    def is_discrete(self):
        space = self.env.action_space
        return hasattr(space, "n")

    def render(self):
        try:
            return self.env.render()
        except Exception:
            try:
                return self.env.render(mode="rgb_array")
            except Exception:
                return np.zeros((64, 64, 3), dtype=np.uint8)
