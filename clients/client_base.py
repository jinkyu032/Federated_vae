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
        self.kl_weight = cfg.kl_weight

    def train(self, local_epochs):

        loss_meter = AverageMeter('Loss', ':.2f')
        recon_loss_meter = AverageMeter('Recon Loss', ':.2f')
        kl_loss_meter = AverageMeter('KL Loss', ':.2f')
        codebook_loss_meter = AverageMeter('Codebook Loss', ':.2f')
        commitment_loss_meter = AverageMeter('Commitment Loss', ':.2f')
        #get the set of the target
        unique_values_set = set()
        self.model.train()
        for _ in range(local_epochs):
            for data, target in self.data_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                if self.cfg.vq:
                    recon_batch, codebook_loss, commitment_loss = self.model(data, target)
                    #breakpoint()
                    recon_loss, _ = self.vae_loss(recon_batch, data, reconloss_only=True, reduction = self.cfg.reduction)
                    if self.cfg.reduction == 'mean':
                        codebook_loss = codebook_loss.mean()
                        commitment_loss = commitment_loss.mean()
                    elif self.cfg.reduction == 'sum':
                        codebook_loss = codebook_loss.sum()
                        commitment_loss = commitment_loss.sum()
                    loss = recon_loss + codebook_loss + self.cfg.commitment_weight * commitment_loss

                    codebook_loss_meter.update(codebook_loss.item(), data.size(0))
                    commitment_loss_meter.update(commitment_loss.item(), data.size(0))


                else:
                    recon_batch, mu, log_var, z = self.model(data, target)
                    recon_loss, kl_loss = self.vae_loss(recon_batch, data, mu, log_var, mu_target=self.vae_mu_target, reduction = self.cfg.reduction)
                    loss = recon_loss + self.kl_weight * kl_loss
                    kl_loss_meter.update(kl_loss.item(), data.size(0))
                loss.backward()
                self.optimizer.step()

                loss_meter.update(loss.item(), data.size(0))
                recon_loss_meter.update(recon_loss.item(), data.size(0))
                
                # Convert tensor to numpy array and then to a set
                batch_unique = set(target.cpu().numpy().flatten())
                
                # Update the set with new unique values
                unique_values_set.update(batch_unique)

        print("Target set: ", unique_values_set)


        
        print(f"Training Loss: {loss_meter.avg:.2f}, Recon Loss: {recon_loss_meter.avg:.2f}, KL Loss: {kl_loss_meter.avg:.2f}")
        loss_dict = {
            "train_loss": loss_meter.avg,
            "train_recon_loss": recon_loss_meter.avg,
            "train_kl_loss": kl_loss_meter.avg
        }
        return self.model.state_dict(), loss_dict

    def update_model(self, global_weights):
        self.model.load_state_dict(global_weights)