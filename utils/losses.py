import torch
import torch.nn as nn

__all__ = ['vae_loss', 'compute_loss']

# VAE loss function
def vae_loss(recon_x, x, mu, log_var, mu_target=0):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(2*mu_target + 1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

# Compute loss for a data loader
def compute_loss(model, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            recon_batch, mu, log_var, z = model(data)
            loss = vae_loss(recon_batch, data, mu, log_var)
            total_loss += loss.item()
    return total_loss / len(data_loader.dataset)