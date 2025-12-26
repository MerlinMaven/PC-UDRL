import numpy as np
import torch

from core.pessimists.quantile import QuantileRegressor
from core.pessimists.cvae import CVAEPessimist
from core.pessimists.diffusion import CondDiffusion
from core.data.dataset import OfflineDataset
from config import make_config


def test_quantile_loss_sign():
    m = QuantileRegressor(4, q=0.9)
    x = torch.randn(8, 4)
    y = torch.randn(8)
    pred = m(x)
    loss = m.loss(pred, y)
    assert loss.item() >= 0.0


def test_cvae_shapes():
    m = CVAEPessimist(4)
    s = torch.randn(16, 4)
    r = torch.randn(16)
    l = m.elbo(s, r)
    cap = m.sample_cap(s)
    assert l.item() >= 0.0
    assert cap.shape == (16,)


def test_diffusion_sample():
    m = CondDiffusion(4, timesteps=10)
    s = torch.randn(8, 4)
    cap = m.sample_cap(s, nsamples=4)
    assert cap.shape == (8,)


def test_dataset_rtg_horizon():
    cfg = make_config(1)
    ds = OfflineDataset(cfg)
    class DummyEnv:
        def reset(self):
            return np.zeros(2, dtype=np.float32)
        def sample_action(self):
            return 0
        def step(self, a):
            return np.zeros(2, dtype=np.float32), 1.0, True, {}
    ds.generate(DummyEnv())
    data = ds.load()
    assert "returns" in data and "horizons" in data
