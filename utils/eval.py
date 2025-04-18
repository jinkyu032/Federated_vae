import torch

from utils.visualize import PlotCallback, plot_latent_space, visualize_manifold, analyze_latent_space
from utils.losses import compute_loss
from typing import Dict
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt

def analyze_model(cfg, model, title, data_loaders: Dict[str, DataLoader],prefix=None, *args, **kwargs):
    
    
    mnist_test_loader = data_loaders["mnist_test"]
    fashion_test_loader = data_loaders["fashion_test"]
    #mnist_test_loader = data_loaders["mnist_test"]#data_loaders["mnist_train"]
    #fashion_test_loader = data_loaders["fashion_test"]#data_loaders["fashion_train"]
    #combined_test_loader = data_loaders["combined_train"]
    combined_test_loader = data_loaders["combined_test"]
    result_dict = {}
    model.eval()
    with torch.no_grad():
        plot_callback = PlotCallback(cfg, num_samples=cfg.num_samples, device=cfg.device, *args, **kwargs)
        #mnist_test_loss, mnist_test_recon_loss, mnist_test_kl_loss = (compute_loss(cfg, model, mnist_test_loader, cfg.device))
        compare_model= kwargs.get('compare_model', None)
        mnist_loss_dict = compute_loss(cfg, model, mnist_test_loader, cfg.device, client_idx=0, compare_model=compare_model)
        # mnist_test_loss = mnist_loss_dict['total_loss']
        # mnist_test_recon_loss = mnist_loss_dict['recon_loss']
        # mnist_test_kl_loss = mnist_loss_dict['kl_loss']
        #fashion_test_loss, fashion_test_recon_loss, fashion_test_kl_loss = (compute_loss(cfg, model, fashion_test_loader, cfg.device))
        fashion_loss_dict = compute_loss(cfg, model, fashion_test_loader, cfg.device, client_idx=1, compare_model=compare_model)
        # fashion_test_loss = fashion_loss_dict['total_loss']
        # fashion_test_recon_loss = fashion_loss_dict['recon_loss']
        # fashion_test_kl_loss = fashion_loss_dict['kl_loss']
        #latent_fig = plot_latent_space(model, mnist_test_loader, fashion_test_loader, title, cfg.device)
        mnist_recon_fig = plot_callback(model, mnist_test_loader)
        fashion_recon_fig = plot_callback(model, fashion_test_loader)
        if not cfg.conditional:
            if cfg.client_classifier:
                mnist_manifold_fig = visualize_manifold(model, device=cfg.device, offset=(cfg.mnist_vae_mu_target, cfg.mnist_vae_mu_target), do = cfg.manifold ,client_classifier = True, **{'client_idx': 0})
                fashion_manifold_fig = visualize_manifold(model, device=cfg.device, offset=(cfg.fashion_vae_mu_target, cfg.fashion_vae_mu_target), do = cfg.manifold,client_classifier = True ,**{'client_idx': 1})
                result_dict[prefix + 'mnist_manifold'] = mnist_manifold_fig
                result_dict[prefix + 'fashion_manifold'] = fashion_manifold_fig

            else:
                if cfg.mnist_vae_mu_target==cfg.fashion_vae_mu_target:
                    manifold_fig = visualize_manifold(model, device=cfg.device, offset=(cfg.mnist_vae_mu_target, cfg.mnist_vae_mu_target), do = cfg.manifold, *args, **kwargs)
                    result_dict[prefix + 'manifold'] = manifold_fig   
                else:
                    mnist_manifold_fig = visualize_manifold(model, device=cfg.device, offset=(cfg.mnist_vae_mu_target, cfg.mnist_vae_mu_target),do = cfg.manifold ,*args, **kwargs)
                    fashion_manifold_fig = visualize_manifold(model, device=cfg.device, offset=(cfg.fashion_vae_mu_target, cfg.fashion_vae_mu_target), do = cfg.manifold , *args, **kwargs)
                    result_dict[prefix + 'mnist_manifold:mu_target='+str(cfg.mnist_vae_mu_target)] = mnist_manifold_fig
                    result_dict[prefix + 'fashion_manifold:mu_target='+str(cfg.fashion_vae_mu_target)] = fashion_manifold_fig


        if cfg.analyze_latent_space:
            if cfg.vq:
                print("Cannot Analyze latent space for VQ-VAE")
            else:
                latent_space_analysis_results = analyze_latent_space(cfg, model, combined_test_loader, cfg.device, args, kwargs)
                for key in latent_space_analysis_results.keys():
                    result_dict[prefix + key] = latent_space_analysis_results[key]

    if cfg.oracle:
        oracle_mnist_test_loss = compute_loss(cfg, model, mnist_test_loader, cfg.device, client_idx=0, oracle=True)
        oracle_fashion_test_loss = compute_loss(cfg, model, fashion_test_loader, cfg.device, client_idx=1, oracle=True)
        for key in oracle_mnist_test_loss.keys():
            result_dict[prefix + 'oracle_mnist_test_' + key] = oracle_mnist_test_loss[key]
        for key in oracle_fashion_test_loss.keys():
            result_dict[prefix + 'oracle_fashion_test_' + key] = oracle_fashion_test_loss[key]


    #result_dict[prefix + 'latent_space'] = latent_fig
    result_dict[prefix + 'mnist_reconstructions'] = mnist_recon_fig
    result_dict[prefix + 'fashion_reconstructions'] = fashion_recon_fig
    # #result_dict[prefix + 'manifold_fig'] = manifold_fig
    # result_dict[prefix + 'mnist_test_loss'] = mnist_test_loss
    # result_dict[prefix + 'fashion_test_loss'] = fashion_test_loss
    # result_dict[prefix + 'mnist_test_recon_loss'] = mnist_test_recon_loss
    # result_dict[prefix + 'fashion_test_recon_loss'] = fashion_test_recon_loss
    # result_dict[prefix + 'mnist_test_kl_loss'] = mnist_test_kl_loss
    # result_dict[prefix + 'fashion_test_kl_loss'] = fashion_test_kl_loss

    #update result_dict with MNIST and FashionMNIST data
    for key in mnist_loss_dict.keys():
        result_dict[prefix + 'mnist_test_' + key] = mnist_loss_dict[key]
    for key in fashion_loss_dict.keys():
        result_dict[prefix + 'fashion_test_' + key] = fashion_loss_dict[key]

    if cfg.use_classifier:
        mnist_test_accuracy = mnist_loss_dict['accuracy']
        fashion_test_accuracy = fashion_loss_dict['accuracy']
        result_dict[prefix + 'mnist_test_accuracy'] = mnist_test_accuracy
        result_dict[prefix + 'fashion_test_accuracy'] = fashion_test_accuracy
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


