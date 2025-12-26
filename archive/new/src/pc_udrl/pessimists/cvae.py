from typing import Optional, Any

import torch
import torch.nn as nn
from pc_udrl.utils import get_device


class CVAEPessimist(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int = 16, hidden_dim: int = 256, cfg: Optional[Any] = None):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.device = get_device(cfg) if cfg is not None else torch.device("cpu")
        self.enc = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mu_q = nn.Linear(hidden_dim, latent_dim)
        self.logvar_q = nn.Linear(hidden_dim, latent_dim)
        self.prior = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mu_p = nn.Linear(hidden_dim, latent_dim)
        self.logvar_p = nn.Linear(hidden_dim, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.to(self.device)

    def encode(self, state: torch.Tensor, ret: torch.Tensor):
        state = state.to(self.device)
        ret = ret.to(self.device)
        x = torch.cat([state, ret.unsqueeze(-1)], dim=-1)
        h = self.enc(x)
        mu = self.mu_q(h)
        logvar = self.logvar_q(h)
        return mu, logvar

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, state: torch.Tensor, z: torch.Tensor):
        state = state.to(self.device)
        z = z.to(self.device)
        x = torch.cat([state, z], dim=-1)
        return self.dec(x).squeeze(-1)

    def prior_params(self, state: torch.Tensor):
        state = state.to(self.device)
        h = self.prior(state)
        mu = self.mu_p(h)
        logvar = self.logvar_p(h)
        return mu, logvar

    def elbo(self, state: torch.Tensor, ret: torch.Tensor):
        mu_q, logvar_q = self.encode(state, ret)
        mu_p, logvar_p = self.prior_params(state)
        z = self.reparam(mu_q, logvar_q)
        recon = self.decode(state, z)
        recon_loss = nn.functional.mse_loss(recon, ret.to(self.device))
        kld = 0.5 * torch.sum(
            logvar_p - logvar_q + (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp() - 1,
            dim=-1,
        ).mean()
        return recon_loss + kld

    def sample_cap(self, state: torch.Tensor, nsamples: int = 16, percentile: float = 0.9):
        state = state.to(self.device)
        mu_p, logvar_p = self.prior_params(state)
        caps = []
        for _ in range(nsamples):
            z = self.reparam(mu_p, logvar_p)
            r = self.decode(state, z)
            caps.append(r)
        caps = torch.stack(caps, dim=0)
        k = int(percentile * nsamples) - 1
        k = max(0, min(nsamples - 1, k))
        vals, _ = torch.sort(caps, dim=0)
        return vals[k]
