import os
from typing import Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from pc_udrl.utils import get_device


class OfflineDataset(Dataset):
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.path = os.path.join(cfg.dataset_dir, f"{cfg.env_id}.pt")
        self.device = get_device(cfg)
        self.data = None

    def exists(self):
        return os.path.exists(self.path)

    def generate(self, env: Any) -> None:
        episodes = []
        for _ in range(self.cfg.episodes):
            s = env.reset()
            traj = {"obs": [], "actions": [], "rewards": [], "dones": []}
            for t in range(self.cfg.max_steps):
                a = env.sample_action()
                o2, r, d, _ = env.step(a)
                traj["obs"].append(np.array(s, dtype=np.float32))
                traj["actions"].append(a)
                traj["rewards"].append(r)
                traj["dones"].append(float(d))
                s = o2
                if d:
                    break
            episodes.append(traj)
        data = self._to_tensors(episodes)
        torch.save(data, self.path)
        self.data = data

    def _save_episodes(self, episodes):
        data = self._to_tensors(episodes)
        torch.save(data, self.path)
        self.data = data

    def load(self):
        self.data = torch.load(self.path)
        return self.data

    def _to_tensors(self, episodes: Any):
        obs = []
        actions = []
        rewards = []
        horizons = []
        terminals = []
        returns = []
        gamma = self.cfg.gamma
        for traj in episodes:
            R = 0.0
            rtg = []
            for r in reversed(traj["rewards"]):
                R = r + gamma * R
                rtg.append(R)
            rtg = list(reversed(rtg))
            H = list(reversed(list(range(1, len(traj["rewards"]) + 1))))
            for i in range(len(traj["rewards"])):
                obs.append(traj["obs"][i])
                actions.append(traj["actions"][i])
                rewards.append(traj["rewards"][i])
                terminals.append(traj["dones"][i])
                returns.append(rtg[i])
                horizons.append(H[i])
        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        actions_np = np.array(actions)
        if actions_np.ndim == 1:
            actions = torch.tensor(actions_np, dtype=torch.int64)
        else:
            actions = torch.tensor(actions_np, dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        returns = torch.tensor(np.array(returns), dtype=torch.float32)
        horizons = torch.tensor(np.array(horizons), dtype=torch.float32)
        dones = torch.tensor(np.array(terminals), dtype=torch.float32)
        return {"obs": obs, "actions": actions, "rewards": rewards, "returns": returns, "horizons": horizons, "dones": dones}

    def __len__(self) -> int:
        if self.data is None:
            self.load()
        return int(self.data["obs"].shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.data is None:
            self.load()
        s = self.data["obs"][idx]
        h = self.data["horizons"][idx]
        dr = self.data["returns"][idx]
        a = self.data["actions"][idx]
        return s, h, dr, a

    def split_train_val(self, val_split: float) -> Tuple[Subset, Subset]:
        if self.data is None:
            self.load()
        n = self.__len__()
        n_val = int(max(1, min(n - 1, n * val_split)))
        indices = np.arange(n)
        rng = np.random.RandomState(self.cfg.seed)
        rng.shuffle(indices)
        val_idx = indices[:n_val].tolist()
        train_idx = indices[n_val:].tolist()
        return Subset(self, train_idx), Subset(self, val_idx)
