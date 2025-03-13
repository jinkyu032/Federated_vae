import torch

from utils.visualize import PlotCallback, plot_latent_space, visualize_manifold
from utils.losses import compute_loss
from typing import Dict
from torch.utils.data import DataLoader

def analyze_model(cfg, model, title, data_loaders: Dict[str, DataLoader]):
    
    
    mnist_test_loader = data_loaders["mnist_test"]
    fashion_test_loader = data_loaders["fashion_test"]
    mnist_train_loader = data_loaders["mnist_train"]
    fashion_train_loader = data_loaders["fashion_train"]
    
    model.eval()
    with torch.no_grad():
        plot_callback = PlotCallback(num_samples=cfg.num_samples, device=cfg.device)
        mnist_test_loss_avg = (compute_loss(model, mnist_test_loader, cfg.device))
        fashion_test_loss_avg = (compute_loss(model, fashion_test_loader, cfg.device))
        latent_fig = plot_latent_space(model, mnist_test_loader, fashion_test_loader, title, cfg.device)
        mnist_recon_fig = plot_callback(model, mnist_train_loader)
        fashion_recon_fig = plot_callback(model, fashion_train_loader)
        manifold_fig = visualize_manifold(model, device=cfg.device)
    model.train()

    return latent_fig, mnist_recon_fig, fashion_recon_fig, manifold_fig, mnist_test_loss_avg, fashion_test_loss_avg




