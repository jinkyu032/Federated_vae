from .vae import VAE
from typing import Dict


def get_model(cfg: Dict):
    latent_dim = cfg.latent_dim
    if cfg.model_name == "vae":
        return VAE(latent_dim=latent_dim)
    else:
        raise ValueError(f"Invalid model name: {cfg.model_name}")