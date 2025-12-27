import torch
import torch.nn as nn
from torch.nn import functional as F
from pc_udrl.utils import get_device

class CVAEPessimist(nn.Module):
    """
    Conditional VAE for modeling p(r, h | s).
    Learns a state-conditioned prior p(z|s) and posterior q(z|s, r, h).
    Used to generate feasible (return, horizon) pairs for a given state.
    """
    def __init__(self, state_dim, latent_dim=16, hidden_dim=256, command_dim=2, cfg=None):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.command_dim = command_dim # (horizon, return)
        self.device = get_device(cfg) if cfg else torch.device("cpu")
        
        # Posterior Encoder: q(z | s, c)
        self.posterior_enc = nn.Sequential(
            nn.Linear(state_dim + command_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim) # mu, logvar
        )
        
        # Prior Encoder: p(z | s)
        self.prior_enc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim) # mu, logvar
        )
        
        # Decoder: p(c | s, z)
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, command_dim) 
        )
        
        self.to(self.device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_posterior(self, state, command):
        x = torch.cat([state, command], dim=-1)
        params = self.posterior_enc(x)
        mu, logvar = torch.chunk(params, 2, dim=-1)
        return mu, logvar

    def forward_prior(self, state):
        params = self.prior_enc(state)
        mu, logvar = torch.chunk(params, 2, dim=-1)
        return mu, logvar
        
    def forward_decoder(self, state, z):
        x = torch.cat([state, z], dim=-1)
        return self.decoder(x)

    def elbo(self, state, horizon, desired_return):
        """
        Computes the CVAE Loss (Recon + KL).
        horizon and desired_return are (N,) or (N, 1).
        """
        if horizon.dim() == 1: horizon = horizon.unsqueeze(-1)
        if desired_return.dim() == 1: desired_return = desired_return.unsqueeze(-1)
        
        command = torch.cat([horizon, desired_return], dim=-1)
        
        # 1. Posterior q(z|s, c)
        post_mu, post_logvar = self.forward_posterior(state, command)
        z = self.reparameterize(post_mu, post_logvar)
        
        # 2. Prior p(z|s)
        # Note: We enforce the prior to be state-dependent, so we move KL minimization here.
        prior_mu, prior_logvar = self.forward_prior(state)
        
        # 3. Recon p(c|s, z)
        rec_command = self.forward_decoder(state, z)
        
        # KL between two Gaussians
        # KL(N0, N1) = 0.5 * ( tr(inv(S1)S0) + (m1-m0)^T inv(S1) (m1-m0) - k + ln(det(S1)/det(S0)) )
        # Diagonal assumption simplifies this.
        var_p = torch.exp(prior_logvar)
        var_q = torch.exp(post_logvar)
        
        kl = 0.5 * torch.sum( (var_q / var_p) + (prior_mu - post_mu).pow(2) / var_p - 1 + prior_logvar - post_logvar, dim=1)
        
        # Recon loss (MSE) on both Horizon and Return
        # We might want to weight them? Horizon is ~100-1000, Return ~-100 to 200. 
        # Ideally normalize? But let's assume raw for now as per project style.
        # Actually, horizon diff of 10 vs return diff of 10?
        # Let's trust MSE.
        recon_loss = F.mse_loss(rec_command, command, reduction='none').sum(dim=1)
        
        beta = 0.5 # Weight KL less to focus on reconstruction accuracy? Or standard 1.0.
        
        loss = (recon_loss + beta * kl).mean()
        return loss

    def sample_cap(self, state, nsamples=32, percentile=0.9):
        """
        Returns a 'safe return' scalar for evaluation compatibility.
        Samples N potential outcomes using the Prior p(z|s).
        Returns the percentile-th value of predicted 'return'.
        """
        # state: (1, state_dim) or (B, state_dim) assuming B=1 for eval usually.
        # But let's handle batch.
        B = state.size(0)
        
        # Expand state for nsamples: (B, N, D)
        state_exp = state.unsqueeze(1).expand(B, nsamples, -1).reshape(B * nsamples, -1)
        
        # Sample prior p(z|s)
        prior_mu, prior_logvar = self.forward_prior(state_exp)
        z = self.reparameterize(prior_mu, prior_logvar)
        
        # Decode
        cmds = self.forward_decoder(state_exp, z) # (B*N, 2)
        
        # Extract Returns (index 1)
        returns = cmds[:, 1].reshape(B, nsamples)
        
        # Take percentile
        # quantile function in torch?
        # If percentile=0.9 (90%), we take the high end (optimistic?)
        # Wait, for safety, 'quantile' in Phase 1 meant "tau-th quantile of return distribution".
        # If we predict distribution, we want the "realistic best"?
        # Or "pessimistic"?
        # If we want to be SAFE, we shouldn't promise more than we can likely deliver.
        # So maybe a median or slightly lower?
        # But 'Quantile Regressor' with q=0.9 usually predicts the 90th percentile (optimistic upper bound? or pessimistic lower bound of COST?).
        # In simple Bellman, Q is expected return.
        # distributional RL: Quantile 0.1 is pessimistic (worst case), 0.9 is optimistic (best case).
        # We called it "Pessimism Gap" = Target - Cap.
        # If Cap is 0.1 quantile (worst case), it's very low.
        # If Cap is 0.9 quantile, it's high.
        # In `experiments.rst`: Pessimist Quantile = 0.7 for LunarLander. 
        # So we likely want the 70th percentile of possible returns.
        
        val = torch.quantile(returns, percentile, dim=1)
        return val.unsqueeze(1) # (B, 1)
