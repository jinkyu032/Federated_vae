import torch

from utils.visualize import PlotCallback, plot_latent_space, visualize_manifold
from utils.losses import compute_loss
from typing import Dict
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt

def analyze_model(cfg, model, title, data_loaders: Dict[str, DataLoader],prefix=None):
    
    
    mnist_test_loader = data_loaders["mnist_test"]
    fashion_test_loader = data_loaders["fashion_test"]
    mnist_train_loader = data_loaders["mnist_train"]
    fashion_train_loader = data_loaders["fashion_train"]
    result_dict = {}
    model.eval()
    with torch.no_grad():
        plot_callback = PlotCallback(cfg, num_samples=cfg.num_samples, device=cfg.device)
        mnist_test_loss, mnist_test_recon_loss, mnist_test_kl_loss = (compute_loss(model, mnist_test_loader, cfg.device))
        fashion_test_loss, fashion_test_recon_loss, fashion_test_kl_loss = (compute_loss(model, fashion_test_loader, cfg.device))
        latent_fig = plot_latent_space(model, mnist_test_loader, fashion_test_loader, title, cfg.device)
        mnist_recon_fig = plot_callback(model, mnist_train_loader)
        fashion_recon_fig = plot_callback(model, fashion_train_loader)
        if not cfg.conditional:
            if cfg.mnist_vae_mu_target==cfg.fashion_vae_mu_target:
                manifold_fig = visualize_manifold(model, device=cfg.device, offset=(cfg.mnist_vae_mu_target, cfg.mnist_vae_mu_target) )
                result_dict[prefix + 'manifold'] = manifold_fig
            else:
                mnist_manifold_fig = visualize_manifold(model, device=cfg.device, offset=(cfg.mnist_vae_mu_target, cfg.mnist_vae_mu_target))
                fashion_manifold_fig = visualize_manifold(model, device=cfg.device, offset=(cfg.fashion_vae_mu_target, cfg.fashion_vae_mu_target))
                result_dict[prefix + 'mnist_manifold:mu_target='+str(cfg.mnist_vae_mu_target)] = mnist_manifold_fig
                result_dict[prefix + 'fashion_manifold:mu_target='+str(cfg.fashion_vae_mu_target)] = fashion_manifold_fig



    result_dict[prefix + '_latent_space'] = latent_fig
    result_dict[prefix + 'mnist_reconstructions'] = mnist_recon_fig
    result_dict[prefix + 'fashion_reconstructions'] = fashion_recon_fig
    #result_dict[prefix + 'manifold_fig'] = manifold_fig
    result_dict[prefix + 'mnist_test_loss'] = mnist_test_loss
    result_dict[prefix + 'fashion_test_loss'] = fashion_test_loss
    result_dict[prefix + 'mnist_test_recon_loss'] = mnist_test_recon_loss
    result_dict[prefix + 'fashion_test_recon_loss'] = fashion_test_recon_loss
    result_dict[prefix + 'mnist_test_kl_loss'] = mnist_test_kl_loss
    result_dict[prefix + 'fashion_test_kl_loss'] = fashion_test_kl_loss
    return result_dict
    #return latent_fig, mnist_recon_fig, fashion_recon_fig, manifold_fig, mnist_test_loss_avg, fashion_test_loss_avg



def log_analysis(wandb_results, analysis, figures_to_close):
    #update wandb_results with converted wandb.Image objects from plt.Figure objects
    for key in analysis.keys():
        if type(analysis[key]) == plt.Figure:
            figures_to_close.append(analysis[key])
            wandb_results[key] = wandb.Image(analysis[key])
        else:
            wandb_results[key] = analysis[key]
    return wandb_results, figures_to_close
