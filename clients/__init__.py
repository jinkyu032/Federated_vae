from .client_base import BaseClient
from .client_per import PerEncClient, PerDecClient
from .client_global import GlobalClient

from typing import Dict, Optional, Union
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig

def get_client(cfg: Union[Dict, DictConfig], model: nn.Module, data_loader: Optional[DataLoader]=None, vae_loss_type: Optional[str]=None):
    if isinstance(cfg, DictConfig):
        cfg = cfg.client
        
    if cfg.client_type == "base":
        return BaseClient(cfg, model, data_loader, vae_loss_type)
    elif cfg.client_type == "per_enc":
        return PerEncClient(cfg, model, data_loader, vae_loss_type)
    elif cfg.client_type == "per_dec":
        return PerDecClient(cfg, model, data_loader, vae_loss_type)
    elif cfg.client_type == "global":
        return GlobalClient(cfg, model, data_loader, vae_loss_type)
    else:
        raise ValueError(f"Invalid client type: {cfg.client_type}")
