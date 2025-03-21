from .vae import VAE
from typing import Dict, Union
from omegaconf import DictConfig

def get_model(cfg: Union[Dict, DictConfig]):
    if isinstance(cfg, DictConfig):
        cfg_model = cfg.model
    else:
        cfg_model = cfg
        
    latent_dim = cfg_model.latent_dim
    conditional = cfg_model.conditional
    num_classes = cfg_model.num_total_classes
    sample_p = cfg.sample_p
    if cfg_model.model_name == "vae":
        return VAE(latent_dim=latent_dim, conditional=conditional, num_classes=num_classes, sample_p=sample_p)
    else:
        raise ValueError(f"Invalid model name: {cfg_model.model_name}")
