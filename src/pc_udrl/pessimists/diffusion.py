import torch
import torch.nn as nn
import numpy as np
from pc_udrl.utils import get_device

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class CondDiffusion(nn.Module):
    """
    Conditional Diffusion Model for Low-Dimensional Command Space (Horizon, Return).
    Models p(command | state).
    """
    def __init__(self, state_dim, hidden_dim=256, command_dim=2, cfg=None, timesteps=100):
        super().__init__()
        self.state_dim = state_dim
        self.command_dim = command_dim
        self.hidden_dim = hidden_dim
        self.timesteps = timesteps
        self.device = get_device(cfg) if cfg else torch.device("cpu")
        
        # Noise Schedule (Linear)
        beta_start = 0.0001
        beta_end = 0.02
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(self.device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Denoiser Network: (State, NoisyCommand, Time) -> Noise
        # MLP based since dimension is small (2)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.input_mlp = nn.Sequential(
            nn.Linear(state_dim + command_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, command_dim)
        )
        
        self.to(self.device)

    def forward(self, state, command, t):
        """
        Predict noise epsilon given (state, noised_command, t)
        """
        t_emb = self.time_mlp(t)
        x_input = torch.cat([state, command], dim=-1)
        x_emb = self.input_mlp(x_input)
        
        # Condition via concatenation
        h = torch.cat([x_emb, t_emb], dim=-1)
        return self.output_mlp(h)

    def loss(self, state, command):
        """
        Training Loss: MSE(epsilon_pred, epsilon)
        """
        B = state.size(0)
        t = torch.randint(0, self.timesteps, (B,), device=self.device).long()
        noise = torch.randn_like(command)
        
        # q_sample: Add noise to get x_t
        # x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * eps
        sqrt_alpha = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        
        x_t = sqrt_alpha * command + sqrt_one_minus_alpha * noise
        
        noise_pred = self.forward(state, x_t, t)
        return nn.functional.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample_cap(self, state, nsamples=32, percentile=0.7):
        """
        Sample valid commands from p(c|s) and return a conservative estimate.
        """
        B = state.size(0)
        # Expand state
        state_exp = state.unsqueeze(1).expand(B, nsamples, -1).reshape(B * nsamples, -1)
        
        # Start from Gaussian Noise
        x = torch.randn(B * nsamples, self.command_dim, device=self.device)
        
        # Reverse Diffusion Process
        for i in reversed(range(self.timesteps)):
            t = torch.full((B * nsamples,), i, device=self.device, dtype=torch.long)
            noise_pred = self.forward(state_exp, x, t)
            
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            # DDPM Update
            # x_{t-1} = 1/sqrt(alpha) * (x_t - (1-alpha)/sqrt(1-alpha_bar) * eps) + sigma * z
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * noise_pred) + torch.sqrt(beta) * noise
            
        # Returns is index 1
        returns = x[:, 1].reshape(B, nsamples)
        
        # Percentile Strategy
        # If we asked for percentile earlier, we used 0.7 (70th).
        # We likely want a value that is "safe" but "high".
        # Median is safer than max.
        val = torch.quantile(returns, percentile, dim=1)
        return val.unsqueeze(1)

    @torch.no_grad()
    def project(self, state, unsafe_command, strength=0.5):
        """
        Project an unsafe command onto the manifold using SDEEdit (Img2Img).
        1. Add noise to unsafe_command corresponding to t = strength * T
        2. Denoise from there conditioned on state.
        This "pulls" the command towards the feasible manifold.
        """
        B = unsafe_command.size(0)
        start_t = int(self.timesteps * strength)
        
        # 1. Forward Diffuse
        t = torch.full((B,), start_t, device=self.device, dtype=torch.long)
        noise = torch.randn_like(unsafe_command)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        
        x = sqrt_alpha * unsafe_command + sqrt_one_minus_alpha * noise
        
        # 2. Reverse Denoise
        for i in reversed(range(start_t + 1)): # +1 to include 0? range is exclusive
             # Actually loop from start_t down to 0
             pass 
             # Logic is tricky to rewrite loop efficiently, let's just copy loop logic
             
        for i in reversed(range(start_t + 1)):
            t_idx = torch.full((B,), i, device=self.device, dtype=torch.long)
            noise_pred = self.forward(state, x, t_idx)
            
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * noise_pred) + torch.sqrt(beta) * noise
            
        return x
