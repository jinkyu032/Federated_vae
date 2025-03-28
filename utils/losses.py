import torch
import torch.nn as nn

__all__ = ['vae_loss', 'compute_loss']

# VAE loss function
def vae_loss(recon_x, x, mu, log_var, mu_target=0, reduction='mean'):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(2*mu_target*mu + 1 + log_var - mu.pow(2) - log_var.exp())

    if reduction == 'mean':
        BCE /= x.size(0)
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
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            if cfg.use_classifier:
                recon_batch, mu, log_var, z, class_output = model(data, return_classfier_output=True)
                _, predicted = torch.max(class_output.data, 1)
                correct += (predicted == target).sum().item()
            else:
                recon_batch, mu, log_var, z = model(data, target)
            recon_loss, kl_loss = vae_loss(recon_batch, data, mu, log_var, mu_target=mu_target)
            total_loss_sum += (recon_loss.item() + cfg.kl_weight * kl_loss.item())
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()

    result = {}
    result['total_loss'] = total_loss_sum / len(data_loader.dataset)
    result['recon_loss'] = recon_loss_sum / len(data_loader.dataset)
    result['kl_loss'] = kl_loss_sum / len(data_loader.dataset)
    if cfg.use_classifier:
        result['accuracy'] = 100 * correct / len(data_loader.dataset)
    return result
