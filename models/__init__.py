from .vae import VAE
from .vaewithclassifier import VAEWithClassifier
from .vaewithclassifier_parallel import VAEWithClassifier_parallel
from .vaewithclassifier_partial import VAEWithClassifier_partial
from .vaewithclassifier_partial_noreuse import VAEWithClassifier_partial_noreuse
from .vqvae import VQVAE
from typing import Dict



def get_model(cfg: Dict):
    latent_dim = cfg.latent_dim
    conditional = cfg.conditional
    num_classes = cfg.num_total_classes
    batch_norm = cfg.batch_norm
    if cfg.model_name == "vae":
        return VAE(latent_dim=latent_dim, conditional=conditional, num_classes=num_classes, batch_norm=batch_norm)
    elif cfg.model_name == "vaewithclassifier":
        return VAEWithClassifier(latent_dim=latent_dim, num_classes=num_classes, batch_norm=batch_norm, cfg = cfg)
    elif cfg.model_name == "vaewithclassifier_parallel":
        return VAEWithClassifier_parallel(latent_dim=latent_dim, num_classes=num_classes, batch_norm=batch_norm, cfg = cfg)
    elif cfg.model_name == "vaewithclassifier_partial":
        return VAEWithClassifier_partial(latent_dim=latent_dim, num_classes=num_classes, batch_norm=batch_norm, cfg = cfg)
    elif cfg.model_name == "vaewithclassifier_partial_noreuse":
        return VAEWithClassifier_partial_noreuse(latent_dim=latent_dim, num_classes=num_classes, batch_norm=batch_norm, cfg = cfg)
    elif cfg.model_name == "vqvae":
        return VQVAE(latent_dim=latent_dim, conditional=conditional, num_classes=num_classes, batch_norm=batch_norm, cfg = cfg)
    else:
        raise ValueError(f"Invalid model name: {cfg.model_name}")