import os
import torch
from torch.utils.data import DataLoader
from pc_udrl.pessimists.quantile import QuantileRegressor
from pc_udrl.pessimists.cvae import CVAEPessimist
from pc_udrl.pessimists.diffusion import CondDiffusion
from pc_udrl.utils import Logger, save_checkpoint, get_device, set_global_seed


def train_pessimist(cfg, dataset):
    set_global_seed(cfg.seed)
    device = get_device(cfg)
    if dataset.data is None:
        dataset.load()
    train_ds, val_ds = dataset.split_train_val(0.2)
    pin = device.type in ("cuda", "mps")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=pin)
    state_dim = dataset.data["obs"].shape[1]
    if cfg.method == "quantile":
        model = QuantileRegressor(state_dim, hidden_dim=cfg.hidden_dim, q=cfg.pessimist_quantile, cfg=cfg)
    elif cfg.method == "cvae":
        model = CVAEPessimist(state_dim, latent_dim=cfg.cvae_latent_dim, hidden_dim=cfg.hidden_dim, cfg=cfg)
    elif cfg.method == "diffusion":
        model = CondDiffusion(state_dim, hidden_dim=cfg.hidden_dim, cfg=cfg, timesteps=100)
    else:
        raise ValueError("unknown method or method not supported in Phase 1 isolated mode")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr_pessimist)
    epochs = getattr(cfg, "epochs", 10)
    log_dir = os.path.join(cfg.runs_dir, f"phase{cfg.phase}")
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(os.path.join(log_dir, "pessimist.csv"), fieldnames=["epoch", "train_loss", "val_loss"], overwrite=True)
    for epoch in range(epochs):
        model.train()
        total_train = 0.0
        n_train = 0
        for s, h, dr, a in train_loader:
            s = s.to(device)
            dr = dr.to(device)
            h = h.to(device) # Make sure h is on device
            if cfg.method == "quantile":
                pred = model(s)
                loss = model.loss(pred, dr)
            elif cfg.method == "cvae":
                loss = model.elbo(s, h, dr)
            elif cfg.method == "diffusion":
                # Force shapes to be safe
                h_in = h.view(-1, 1)
                dr_in = dr.view(-1, 1)
                cmd = torch.cat([h_in, dr_in], dim=-1)
                loss = model.loss(s, cmd)
            else:
                # Fallback or error
                loss = model.loss(s, dr)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_train += float(loss.item()) * s.size(0)
            n_train += s.size(0)
        train_loss = total_train / max(1, n_train)
        model.eval()
        total_val = 0.0
        n_val = 0
        with torch.no_grad():
            for s, h, dr, a in val_loader:
                s = s.to(device)
                dr = dr.to(device)
                h = h.to(device)
                if cfg.method == "quantile":
                    pred = model(s)
                    loss = model.loss(pred, dr)
                elif cfg.method == "cvae":
                    loss = model.elbo(s, h, dr)
                elif cfg.method == "diffusion":
                    # Force shapes to be safe
                    h_in = h.view(-1, 1)
                    dr_in = dr.view(-1, 1)
                    cmd = torch.cat([h_in, dr_in], dim=-1)
                    loss = model.loss(s, cmd)
                else:
                    loss = model.loss(s, dr)
                total_val += float(loss.item()) * s.size(0)
                n_val += s.size(0)
        val_loss = total_val / max(1, n_val)
        logger.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
    save_checkpoint(model, os.path.join(log_dir, "pessimist.pt"))
    return model

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from config import make_config
    from pc_udrl.data.dataset import OfflineDataset
    from pc_udrl.utils import ensure_dirs

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1)
    args = parser.parse_args()

    cfg = make_config(args.phase)
    ensure_dirs(cfg)
    
    dataset = OfflineDataset(cfg)
    train_pessimist(cfg, dataset)
