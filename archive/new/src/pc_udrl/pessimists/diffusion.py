from typing import Optional, Any

import torch
import torch.nn as nn
from pc_udrl.utils import get_device


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(0, torch.log(torch.tensor(10000.0, device=device)), half, device=device)
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class CondDiffusion(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, cfg: Optional[Any] = None, timesteps: int = 100):
        super().__init__()
        self.device = get_device(cfg) if cfg is not None else torch.device("cpu")
        self.timesteps = timesteps
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.betas = torch.linspace(self.beta_start, self.beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.eps_model = nn.Sequential(
            nn.Linear(state_dim + 1 + 64, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.to(self.device)

    def _schedule_to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)

    def predict_noise(self, state: torch.Tensor, t: torch.Tensor, noisy_ret: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device)
        noisy_ret = noisy_ret.to(self.device)
        t = t.to(self.device)
        t_embed = sinusoidal_time_embedding(t.float(), 64)
        x = torch.cat([state, noisy_ret.unsqueeze(-1), t_embed], dim=-1)
        return self.eps_model(x).squeeze(-1)

    def loss(self, state: torch.Tensor, ret: torch.Tensor) -> torch.Tensor:
        device = self.device
        self._schedule_to(device)
        B = state.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=device)
        alpha_cum = self.alphas_cumprod[t]
        eps = torch.randn_like(ret)
        noisy = torch.sqrt(alpha_cum) * ret + torch.sqrt(1 - alpha_cum) * eps
        pred_eps = self.predict_noise(state, t, noisy)
        return nn.functional.mse_loss(pred_eps, eps)

    def sample_cap(self, state: torch.Tensor, nsamples: int = 16, percentile: float = 0.9) -> torch.Tensor:
        device = self.device
        self._schedule_to(device)
        B = state.shape[0]
        caps = []
        for _ in range(nsamples):
            x = torch.randn(B, device=device)
            for t in reversed(range(self.timesteps)):
                t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
                pred_eps = self.predict_noise(state, t_tensor, x)
                alpha_t = self.alphas[t]
                alpha_cum_t = self.alphas_cumprod[t]
                beta_t = self.betas[t]
                x = (1 / torch.sqrt(self.alphas[t])) * (x - beta_t / torch.sqrt(1 - alpha_cum_t) * pred_eps)
                if t > 0:
                    x = x + torch.sqrt(beta_t) * torch.randn_like(x)
            caps.append(x)
        caps = torch.stack(caps, dim=0)
        k = int(percentile * nsamples) - 1
        k = max(0, min(nsamples - 1, k))
        vals, _ = torch.sort(caps, dim=0)
        return vals[k]

