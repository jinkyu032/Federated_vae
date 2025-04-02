

from .vae import VAE
from .vaewithclassifier import VAEWithClassifier
from .vaewithclassifier_parallel import VAEWithClassifier_parallel
from .vaewithclassifier_partial import VAEWithClassifier_partial
from .vaewithclassifier_partial_noreuse import VAEWithClassifier_partial_noreuse
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
    batch_norm = cfg_model.batch_norm
    model_name = cfg_model.model_name
    
    if model_name == "vae":
        return VAE(latent_dim=latent_dim, conditional=conditional, num_classes=num_classes, batch_norm=batch_norm)
    elif model_name == "vaewithclassifier":
        return VAEWithClassifier(latent_dim=latent_dim, num_classes=num_classes, batch_norm=batch_norm, cfg = cfg)
    elif model_name == "vaewithclassifier_parallel":
        return VAEWithClassifier_parallel(latent_dim=latent_dim, num_classes=num_classes, batch_norm=batch_norm, cfg = cfg)
    elif model_name == "vaewithclassifier_partial":
        return VAEWithClassifier_partial(latent_dim=latent_dim, num_classes=num_classes, batch_norm=batch_norm, cfg = cfg)
    elif model_name == "vaewithclassifier_partial_noreuse":
        return VAEWithClassifier_partial_noreuse(latent_dim=latent_dim, num_classes=num_classes, batch_norm=batch_norm, cfg = cfg)
    else:
        raise ValueError(f"Invalid model name: {cfg_model.model_name}")
