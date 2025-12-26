import os
import csv
import random
from typing import Optional, Dict, Any, Iterable

import numpy as np
import torch


def ensure_dirs(cfg):
    os.makedirs(cfg.dataset_dir, exist_ok=True)
    os.makedirs(cfg.runs_dir, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Logger:
    def __init__(self, csv_path: str, fieldnames: Optional[Iterable[str]] = None, overwrite: bool = False):
        self.csv_path = csv_path
        self.fieldnames = list(fieldnames) if fieldnames is not None else None
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        
        mode = "w" if overwrite or not os.path.exists(self.csv_path) else "a"
        if mode == "w" and self.fieldnames is not None:
             with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, metrics: Dict[str, Any]) -> None:
        print(" ".join(f"{k}={metrics[k]}" for k in metrics))
        if self.fieldnames is None:
            self.fieldnames = list(metrics.keys())
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
        row = {k: metrics.get(k, "") for k in self.fieldnames}
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str, map_location: Optional[torch.device] = None, strict: bool = True) -> torch.nn.Module:
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state, strict=strict)
    return model


def get_device(cfg: Any) -> torch.device:
    d = getattr(cfg, "device", "cpu")
    if isinstance(d, str):
        if d in ("cuda", "gpu") or d.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device("cuda") if d == "gpu" else torch.device(d)
            return torch.device("cpu")
        if d == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device("cpu")
    return torch.device("cpu")

