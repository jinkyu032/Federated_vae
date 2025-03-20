from utils.losses import vae_loss
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
__all__ = ['BaseClient']

# Client class for federated learning
class BaseClient:
    def __init__(self, cfg: Dict, model: nn.Module, data_loader: Optional[DataLoader]=None, vae_mu_target: Optional[int]=None):
        self.cfg = cfg
        self.device = cfg.device
        self.data_loader = data_loader
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.vae_loss = vae_loss
        self.vae_mu_target = vae_mu_target

    def train(self, local_epochs):
        # self.model.load_state_dict(global_weights)
        self.model.train()
        for _ in range(local_epochs):
            for data, target in self.data_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, log_var, z = self.model(data, target)
                loss = self.vae_loss(recon_batch, data, mu, log_var, mu_target=self.vae_mu_target)
                loss.backward()
                self.optimizer.step()
        return self.model.state_dict()

    def update_model(self, global_weights):
        self.model.load_state_dict(global_weights)