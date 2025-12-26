import torch
import torch.nn as nn


class UDRLAgent(nn.Module):
    """
    Upside-Down Reinforcement Learning (UDRL) Agent.

    This agent predicts the optimal action given a state and a command (horizon, desired_return).
    It learns via supervised learning to mimic trajectories that achieved high returns, effectively
    inverting the RL problem (mapping Command -> Action).

    Args:
        state_dim (int): Dimension of the observation space.
        action_dim (int): Dimension of the action space.
        discrete (bool): Whether the action space is discrete (True) or continuous (False).
        hidden_dim (int, optional): Size of the hidden layers. Defaults to 256.
    """
    def __init__(self, state_dim, action_dim, discrete, hidden_dim=256):
        super().__init__()
        self.discrete = discrete
        self.state_enc = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.cmd_enc = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim))

    def forward(self, state, horizon, desired_return):
        """
        Forward pass for the UDRL agent.

        Args:
            state (torch.Tensor): Current state observation.
            horizon (torch.Tensor): Remaining time steps/horizon.
            desired_return (torch.Tensor): Desired return-to-go.

        Returns:
            torch.Tensor: Predicted action logits (discrete) or values (continuous).
        """
        s = self.state_enc(state)
        c = self.cmd_enc(torch.stack([horizon, desired_return], dim=-1))
        x = torch.cat([s, c], dim=-1)
        out = self.head(x)
        if self.discrete:
            return out
        return torch.tanh(out)

    def act(self, state, horizon, desired_return):
        """
        Selects an action for the given state and command during inference.

        Args:
            state (torch.Tensor): Current state.
            horizon (torch.Tensor): Desired horizon.
            desired_return (torch.Tensor): Desired return.

        Returns:
            torch.Tensor or int: Selected action (index for discrete, tensor for continuous).
        """
        with torch.no_grad():
            logits = self.forward(state, horizon, desired_return)
            if self.discrete:
                return torch.argmax(logits, dim=-1)
            return logits
