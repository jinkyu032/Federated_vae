from .vae import VAE
from typing import Dict, Union
from omegaconf import DictConfig

def get_model(cfg: Union[Dict, DictConfig]):
    latent_dim = cfg.model.latent_dim
    conditional = cfg.model.conditional
    num_classes = cfg.model.num_total_classes
    sample_p = cfg.sample_p
    if cfg.model.model_name == "vae":
        return VAE(latent_dim=latent_dim, conditional=conditional, num_classes=num_classes, sample_p=sample_p)
    else:
        raise ValueError(f"Invalid model name: {cfg.model.model_name}")
