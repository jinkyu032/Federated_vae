import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import gc # Garbage collector for explicit memory management
from sklearn.feature_selection import mutual_info_classif
__all__ = ['vae_loss', 'compute_loss']

# VAE loss function
def vae_loss(recon_x, x, mu = None, log_var = None, mu_target=0, reduction='mean', reconloss_only=False):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = None
    if not reconloss_only:
        KLD = -0.5 * torch.sum(2*mu_target*mu + 1 + log_var - mu.pow(2) - log_var.exp() - mu_target*mu_target)

    if reduction == 'mean':
        BCE /= x.size(0)
        if not reconloss_only:
            KLD /= x.size(0)

    return BCE, KLD

# Compute loss for a data loader
def compute_loss(cfg, model, data_loader, device, mu_target=0, reduction='sum'):
    model.eval()
    model.to(device)
    total_loss_sum = 0
    recon_loss_sum = 0
    kl_loss_sum = 0
    correct = 0
    codebook_loss_sum = 0
    commitment_loss_sum = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            if cfg.vq:
                recon_batch, codebook_loss, commitment_loss = model(data, target)
                recon_loss, _ = vae_loss(recon_batch, data, reconloss_only=True, reduction=reduction)
                if reduction == 'mean':
                    codebook_loss = codebook_loss.mean()
                    commitment_loss = commitment_loss.mean()
                elif reduction == 'sum':
                    codebook_loss = codebook_loss.sum()
                    commitment_loss = commitment_loss.sum()
                total_loss_sum += (recon_loss.item() + codebook_loss.item() + cfg.commitment_weight * commitment_loss.item())
                recon_loss_sum += recon_loss.item()
                codebook_loss_sum += codebook_loss.item()
                commitment_loss_sum += commitment_loss.item()
            else:
                if cfg.use_classifier:
                    recon_batch, mu, log_var, z, class_output = model(data, return_classfier_output=True)
                    _, predicted = torch.max(class_output.data, 1)
                    correct += (predicted == target).sum().item()
                else:
                    recon_batch, mu, log_var, z = model(data, target)
                recon_loss, kl_loss = vae_loss(recon_batch, data, mu, log_var, mu_target=mu_target, reduction = 'sum')
                total_loss_sum += (recon_loss.item() + cfg.kl_weight * kl_loss.item())
                recon_loss_sum += recon_loss.item()
                kl_loss_sum += kl_loss.item()

    result = {}
    result['total_loss'] = total_loss_sum / len(data_loader.dataset)
    result['recon_loss'] = recon_loss_sum / len(data_loader.dataset)
    if cfg.vq:
        result['codebook_loss'] = codebook_loss_sum / len(data_loader.dataset)
        result['commitment_loss'] = commitment_loss_sum / len(data_loader.dataset)
    else:
        result['kl_loss'] = kl_loss_sum / len(data_loader.dataset)
    if cfg.use_classifier:
        result['accuracy'] = 100 * correct / len(data_loader.dataset)
    return result



