from utils.losses import vae_loss
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from utils.logging_utils import AverageMeter
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

        loss_meter = AverageMeter('Loss', ':.2f')
        recon_loss_meter = AverageMeter('Recon Loss', ':.2f')
        kl_loss_meter = AverageMeter('KL Loss', ':.2f')

        self.model.train()
        for _ in range(local_epochs):
            for data, target in self.data_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, log_var, z = self.model(data, target)
                recon_loss, kl_loss = self.vae_loss(recon_batch, data, mu, log_var, mu_target=self.vae_mu_target)
                loss = recon_loss + kl_loss
                loss.backward()
                self.optimizer.step()

                loss_meter.update(loss.item(), data.size(0))
                recon_loss_meter.update(recon_loss.item(), data.size(0))
                kl_loss_meter.update(kl_loss.item(), data.size(0))

        
        print(f"Training Loss: {loss_meter.avg:.2f}, Recon Loss: {recon_loss_meter.avg:.2f}, KL Loss: {kl_loss_meter.avg:.2f}")
        loss_dict = {
            "train_loss": loss_meter.avg,
            "train_recon_loss": recon_loss_meter.avg,
            "train_kl_loss": kl_loss_meter.avg
        }
        return self.model.state_dict(), loss_dict

    def update_model(self, global_weights):
        self.model.load_state_dict(global_weights)