from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from omegaconf import DictConfig

from clients.client_base import BaseClient


__all__ = ['PerEncClient', 'PerDecClient']

# Client class for federated learning
class PerEncClient(BaseClient):
    def __init__(self, Union[Dict, DictConfig], model: nn.Module, data_loader: Optional[DataLoader]=None, vae_mu_target: Optional[int]=None):
        super(PerEncClient, self).__init__(cfg, model, data_loader, vae_mu_target)

    def update_model(self, global_weights):
        model_dict = self.model.state_dict()
        global_dict = {k: v for k, v in global_weights.items() if 'decoder' in k}
        model_dict.update(global_dict)
        self.model.load_state_dict(model_dict, strict=False)
        print(f"Updated decoder weights for client")

# Client class for federated learning
class PerDecClient(BaseClient):
    def __init__(self, Union[Dict, DictConfig], model: nn.Module, data_loader: Optional[DataLoader]=None, vae_mu_target: Optional[int]=None):
        super(PerDecClient, self).__init__(cfg, model, data_loader, vae_mu_target)

    def update_model(self, global_weights):
        model_dict = self.model.state_dict()
        global_dict = {k: v for k, v in global_weights.items() if 'encoder' in k}
        model_dict.update(global_dict)
        self.model.load_state_dict(model_dict, strict=False)
        print(f"Updated encoder weights for client")
