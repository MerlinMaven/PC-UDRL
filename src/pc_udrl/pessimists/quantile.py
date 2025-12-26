from typing import Optional, Any

import torch
import torch.nn as nn
from pc_udrl.utils import get_device


class QuantileRegressor(nn.Module):
    """
    Pessimistic Oracle using Quantile Regression.

    Predicts the tau-th quantile of the return distribution for a given state.
    Used to clamp user commands to a realistic/safe range, implementing the
    concept of "Pessimistic Projection".

    Args:
        state_dim (int): Dimension of the observation space.
        hidden_dim (int, optional): Size of the hidden layers. Defaults to 256.
        q (float, optional): Quantile to predict (e.g., 0.9 for 90th percentile). Defaults to 0.9.
        cfg (Any, optional): Configuration object containing device settings.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 256, q: float = 0.9, cfg: Optional[Any] = None):
        super().__init__()
        self.q: float = q
        self.device: torch.device = get_device(cfg) if cfg is not None else torch.device("cpu")
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predicts the return quantile for the given state.

        Args:
            state (torch.Tensor): State observation.

        Returns:
            torch.Tensor: Predicted return value (scalar).
        """
        state = state.to(next(self.parameters()).device)
        return self.net(state).squeeze(-1)

    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the Quantile Huber Loss (Pinball Loss).

        Args:
            pred (torch.Tensor): Predicted quantiles.
            target (torch.Tensor): Actual returns (targets).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        device = pred.device
        e = target.to(device) - pred
        return torch.maximum(self.q * e, (self.q - 1) * e).mean()
