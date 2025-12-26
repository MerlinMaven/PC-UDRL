import os
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import make_config
from core.envs.gym_wrappers import make_env
from core.data.dataset import OfflineDataset
from core.utils import set_global_seed, get_device


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(state)
        mu = torch.tanh(self.mu_head(h))
        log_std = self.log_std.expand_as(mu)
        return mu, log_std

    def sample(self, state: torch.Tensor):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        action = mu + eps * std
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = -0.5 * (((action - mu) / (std + 1e-8)) ** 2 + 2 * log_std + math.log(2 * math.pi)).sum(dim=-1)
        return action, log_prob


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


@dataclass
class PPOCfg:
    steps: int = 20000
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    epochs: int = 10
    batch_size: int = 256
    max_steps_per_ep: int = 1000


def collect(env, policy, value, device, steps, max_steps_per_ep, gamma, lam):
    states, actions, rewards, dones, values, logps = [], [], [], [], [], []
    s = env.reset()
    ep_steps = 0
    for _ in range(steps):
        st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a, logp = policy.sample(st)
            v = value(st)
        a_np = a.squeeze(0).detach().cpu().numpy()
        o2, r, d, _ = env.step(a_np)
        states.append(st.squeeze(0))
        actions.append(torch.tensor(a_np, dtype=torch.float32, device=device))
        rewards.append(torch.tensor(r, dtype=torch.float32, device=device))
        dones.append(torch.tensor(d, dtype=torch.float32, device=device))
        values.append(v.squeeze(0))
        logps.append(logp.squeeze(0))
        s = o2
        ep_steps += 1
        if d or ep_steps >= max_steps_per_ep:
            s = env.reset()
            ep_steps = 0
    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = torch.stack(rewards)
    dones = torch.stack(dones)
    values = torch.stack(values)
    logps = torch.stack(logps)
    adv = torch.zeros_like(rewards)
    ret = torch.zeros_like(rewards)
    last_gae = 0.0
    last_ret = 0.0
    for t in reversed(range(steps)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * last_ret * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        adv[t] = last_gae
        last_ret = values[t] + adv[t]
        ret[t] = last_ret
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return states, actions, logps, values, adv, ret


def train_ppo(env, state_dim, action_dim, device, cfg: PPOCfg):
    policy = GaussianPolicy(state_dim, action_dim).to(device)
    value = ValueNet(state_dim).to(device)
    opt_pi = optim.Adam(policy.parameters(), lr=cfg.lr)
    opt_v = optim.Adam(value.parameters(), lr=cfg.lr)
    total = 0
    while total < cfg.steps:
        steps = min(4096, cfg.steps - total)
        states, actions, old_logps, old_values, adv, ret = collect(
            env, policy, value, device, steps, cfg.max_steps_per_ep, cfg.gamma, cfg.lam
        )
        total += steps
        idx = torch.randperm(steps, device=device)
        for _ in range(cfg.epochs):
            for start in range(0, steps, cfg.batch_size):
                j = idx[start:start + cfg.batch_size]
                s_b = states[j]
                a_b = actions[j]
                adv_b = adv[j]
                ret_b = ret[j]
                old_logp_b = old_logps[j]
                mu, log_std = policy.forward(s_b)
                std = log_std.exp()
                logp = -0.5 * (((a_b - mu) / (std + 1e-8)) ** 2 + 2 * log_std + math.log(2 * math.pi)).sum(dim=-1)
                ratio = torch.exp(logp - old_logp_b)
                obj1 = ratio * adv_b
                obj2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * adv_b
                loss_pi = -(torch.min(obj1, obj2)).mean()
                opt_pi.zero_grad()
                loss_pi.backward()
                opt_pi.step()
                v_b = value(s_b)
                loss_v = nn.functional.mse_loss(v_b, ret_b)
                opt_v.zero_grad()
                loss_v.backward()
                opt_v.step()
    return policy


def collect_dataset(env, policy, device, episodes: int, max_steps: int):
    ep_list = []
    for _ in range(episodes):
        s = env.reset()
        traj = {"obs": [], "actions": [], "rewards": []}
        for t in range(max_steps):
            st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a, _ = policy.sample(st)
            a_np = a.squeeze(0).detach().cpu().numpy()
            o2, r, d, _ = env.step(a_np)
            traj["obs"].append(np.array(s, dtype=np.float32))
            traj["actions"].append(a_np.astype(np.float32))
            traj["rewards"].append(float(r))
            s = o2
            if d:
                break
        ep_list.append(traj)
    return ep_list


def main(phase: int):
    cfg = make_config(phase)
    assert cfg.env_id == "LunarLanderContinuous-v2", "Baseline PPO is intended for phases 2/3"
    set_global_seed(cfg.seed)
    device = get_device(cfg)
    env = make_env(cfg.env_id, cfg.seed)
    state_dim = env.observation_space[0]
    action_dim = env.action_space if isinstance(env.action_space, int) else int(env.action_space)
    ppo_cfg = PPOCfg(steps=10000, gamma=cfg.gamma, lam=0.95, lr=cfg.lr_agent, epochs=5, batch_size=cfg.batch_size, max_steps_per_ep=cfg.max_steps)
    policy = train_ppo(env, state_dim, action_dim, device, ppo_cfg)
    episodes = collect_dataset(env, policy, device, episodes=cfg.episodes, max_steps=cfg.max_steps)
    dataset = OfflineDataset(cfg)
    data = dataset._to_tensors(episodes)
    os.makedirs(cfg.dataset_dir, exist_ok=True)
    torch.save(data, dataset.path)


if __name__ == "__main__":
    main(2)
