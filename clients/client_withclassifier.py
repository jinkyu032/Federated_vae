import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
from utils.logging_utils import AverageMeter
from utils.losses import vae_loss  # Assuming you have this loss function
from torch.nn import CrossEntropyLoss

__all__ = ['VAEClassifierClient']

# Client class for federated learning with VAE and Classifier
class VAEClassifierClient:
    def __init__(self, cfg: Dict, model: nn.Module, data_loader: Optional[DataLoader] = None, vae_mu_target: Optional[int] = None):
        self.cfg = cfg
        self.device = cfg.device
        self.data_loader = data_loader
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.vae_loss = vae_loss
        self.vae_mu_target = vae_mu_target
        self.kl_weight = cfg.kl_weight
        # if cfg.problabelfeatures:
        #     self.classifier_loss = nn.NLLLoss(reduction=cfg.reduction)  # Loss for classifier
        # else:
        self.classifier_loss = CrossEntropyLoss(reduction=cfg.reduction)  # Loss for classifier
        self.use_classifier = cfg.use_classifier  # Set to False if you don't want to use classifier

    def train(self, local_epochs):

        loss_meter = AverageMeter('Loss', ':.2f')
        recon_loss_meter = AverageMeter('Recon Loss', ':.2f')
        kl_loss_meter = AverageMeter('KL Loss', ':.2f')
        classifier_loss_meter = AverageMeter('Classifier Loss', ':.2f')

        self.model.train()
        for _ in range(local_epochs):
            for data, target in self.data_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()

                recon_batch, mu, log_var, z, class_output = self.model(data, return_classfier_output=self.use_classifier)

                # VAE Loss
                recon_loss, kl_loss = self.vae_loss(recon_batch, data, mu, log_var, mu_target=self.vae_mu_target, reduction = self.cfg.reduction)
                vae_total_loss = recon_loss + self.kl_weight * kl_loss

                # Classifier Loss
                classifier_loss = self.classifier_loss(class_output/self.cfg.temperature, target)

                # Total Loss
                loss = vae_total_loss + classifier_loss  # Joint training, adjust weight as needed

                loss.backward()
                self.optimizer.step()

                loss_meter.update(loss.item(), data.size(0))
                recon_loss_meter.update(recon_loss.item(), data.size(0))
                kl_loss_meter.update(kl_loss.item(), data.size(0))
                classifier_loss_meter.update(classifier_loss.item(), data.size(0))

        print(f"Training Loss: {loss_meter.avg:.2f}, Recon Loss: {recon_loss_meter.avg:.2f}, KL Loss: {kl_loss_meter.avg:.2f}, Classifier Loss: {classifier_loss_meter.avg:.2f}")
        loss_dict = {
            "train_loss": loss_meter.avg,
            "train_recon_loss": recon_loss_meter.avg,
            "train_kl_loss": kl_loss_meter.avg,
            "train_classifier_loss": classifier_loss_meter.avg
        }
        return self.model.state_dict(), loss_dict

    def update_model(self, global_weights):
        self.model.load_state_dict(global_weights)