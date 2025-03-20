from .vae import VAE
from typing import Dict


def get_model(cfg: Dict):
    latent_dim = cfg.latent_dim
    conditional = cfg.conditional
    num_classes = cfg.num_total_classes
    sample_p = cfg.sample_p
    if cfg.model_name == "vae":
        return VAE(latent_dim=latent_dim, conditional=conditional, num_classes=num_classes, sample_p=sample_p)
    else:
        raise ValueError(f"Invalid model name: {cfg.model_name}")
