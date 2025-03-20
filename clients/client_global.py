from utils.losses import vae_loss
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from copy import deepcopy
from utils.logging_utils import AverageMeter
import torch
__all__ = ['GlobalClient']

# Client class for federated learning
class GlobalClient:
    def __init__(self, cfg: Dict, model: nn.Module, data_loader: Optional[DataLoader]=None, vae_mu_target: Optional[int]=0, vae_sigma_target: Optional[int]=1):
        self.cfg = cfg
        self.device = cfg.device
        self.data_loader = data_loader
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.vae_loss = vae_loss
        self.vae_mu_target = vae_mu_target
        self.vae_sigma_target = vae_sigma_target
        self.iterations = 0

    
    def train(self, local_epochs):

        loss_meter = AverageMeter('Loss', ':.2f')
        recon_loss_meter = AverageMeter('Recon Loss', ':.2f')
        kl_loss_meter = AverageMeter('KL Loss', ':.2f')
        # self.model.load_state_dict(global_weights)
        self.model.train()
        self.mnist_client_model = self.mnist_client_model.to(self.device)
        self.fashion_client_model = self.fashion_client_model.to(self.device)
        for _ in range(local_epochs):
            for i in range(self.iterations):
                # Sample from normal gaussian distribution with vae_mu_target and vae_sigma_target
                batch_size = self.cfg.batch_size
                latent_dim = self.cfg.latent_dim
                
                # Sample from normal distribution
                z = torch.normal(mean=self.vae_mu_target, std=self.vae_sigma_target, size=(batch_size, latent_dim))
                z = z.to(self.device)

                mnist_data = self.mnist_client_model.decoder_forward(z).detach()
                mnist_data = mnist_data.reshape(batch_size, -1, 28, 28)
                fashion_data = self.fashion_client_model.decoder_forward(z).detach()
                fashion_data = fashion_data.reshape(batch_size, -1, 28, 28)

                data = torch.cat((mnist_data, fashion_data), dim=0)
                
                self.optimizer.zero_grad()
                recon_batch, mu, log_var, z = self.model(data)
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
        # wandb.log(loss_dict, step=self.iterations)
        return self.model.state_dict(), loss_dict

    def get_local_models(self, mnist_client_model_state_dict, fashion_client_model_state_dict):
        
        self.mnist_client_model = deepcopy(self.model)
        print(self.mnist_client_model.load_state_dict(mnist_client_model_state_dict))
        self.fashion_client_model = deepcopy(self.model)
        print(self.fashion_client_model.load_state_dict(fashion_client_model_state_dict))

        self.mnist_client_model.eval()
        self.fashion_client_model.eval()
        
        