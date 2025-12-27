from dataclasses import dataclass


@dataclass
class Config:
    phase: int = 1
    env_id: str = "GridWorld"
    method: str = "quantile"
    seed: int = 0
    device: str = "cpu"
    dataset_dir: str = "data"
    runs_dir: str = "runs"
    output_dir: str = "outputs"
    episodes: int = 200 # Reduced for benchmark speed
    max_steps: int = 200
    gamma: float = 0.99
    batch_size: int = 256
    lr_agent: float = 3e-4
    lr_pessimist: float = 3e-4
    hidden_dim: int = 256
    pessimist_quantile: float = 0.9
    cvae_latent_dim: int = 16
    eval_episodes: int = 20
    eval_interval: int = 2 # Run eval every N epochs during training
    epochs: int = 30 # Reduced for benchmark speed


def make_config(phase: int) -> Config:
    if phase == 1:
        # Phase 1: GridWorld POC
        return Config(phase=1, env_id="GridWorld", method="quantile", max_steps=100)
    if phase == 2:
        # Phase 2: LunarLander Validation + Baselines
        return Config(phase=2, env_id="LunarLanderContinuous-v3", method="quantile", episodes=1000, max_steps=1000, pessimist_quantile=0.7)
    if phase == 3:
        # Phase 3: Advanced Pessimism (CVAE/Diffusion)
        # Allows easy switching: method="cvae" or "diffusion"
        return Config(phase=3, env_id="LunarLanderContinuous-v3", method="cvae", episodes=1000)
    if phase == 4:
        # Phase 4: Diffusion
        return Config(phase=4, env_id="LunarLanderContinuous-v3", method="diffusion", episodes=1000, max_steps=1000)
    return Config()

