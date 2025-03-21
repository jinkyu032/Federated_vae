import hydra
import torch
import wandb
import matplotlib.pyplot as plt
import gc
import omegaconf
from omegaconf import DictConfig, OmegaConf
from typing import Dict
from torch.utils.data import DataLoader
import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm

from utils.losses import compute_loss
from utils.eval import analyze_model, log_analysis
from utils.data import get_dataloaders
from clients import get_client
from servers import get_server
from models import get_model
from utils.visualize import plot_latent_per_client

# Federated training
def train_federated(cfg, data_loaders: Dict[str, DataLoader], model: nn.Module):
    wandb_results = {}

    # Get Clients
    MNISTClient = get_client(cfg, deepcopy(model), data_loaders["mnist_train"], cfg.mnist_vae_mu_target)
    FashionClient = get_client(cfg, deepcopy(model), data_loaders["fashion_train"], cfg.fashion_vae_mu_target)

    # Get Server
    server = get_server(cfg, deepcopy(model))

    num_rounds = cfg.num_rounds
    local_epochs = cfg.local_epochs
   
    for round_num in tqdm(range(num_rounds)):
        # Train Clients
        client_weights = [
            MNISTClient.train(local_epochs),
            FashionClient.train(local_epochs)
        ]

        # Aggregate Client Weights
        server.aggregate(client_weights)
        
        ## Eval && analysis
        figures_to_close = []
        
        if cfg.analyze_local_models_before_update:
            # Analyze MNISTClient.model
            MNIST_analysis = analyze_model(cfg, MNISTClient.model, f"Client 1 Round {round_num+1}", data_loaders=data_loaders, prefix="MNISTClient_")
            wandb_results, figures_to_close = log_analysis(wandb_results, MNIST_analysis, figures_to_close)
            
            # Analyze FashionClient.model
            Fashion_analysis = analyze_model(cfg, FashionClient.model, f"Client 2 Round {round_num+1}", data_loaders=data_loaders, prefix="FashionClient_")
            wandb_results, figures_to_close = log_analysis(wandb_results, Fashion_analysis, figures_to_close)

        # Update Client Models
        MNISTClient.update_model(server.global_model.state_dict())
        FashionClient.update_model(server.global_model.state_dict())
        
        mnist_train_loss_avg, _, _, _ = compute_loss(MNISTClient.model, data_loaders["mnist_train"], cfg.device, mu_target=cfg.mnist_vae_mu_target, alpha=cfg.alpha) 
        fashion_train_loss_avg, _, _, _ = compute_loss(FashionClient.model, data_loaders["fashion_train"], cfg.device, mu_target=cfg.fashion_vae_mu_target, alpha=cfg.alpha)

        wandb_results.update({
            "MNIST_train_loss": mnist_train_loss_avg,
            "Fashion_train_loss": fashion_train_loss_avg,
        })
        
        if not cfg.analyze_local_models_before_update:
            # Analyze MNISTClient.model
            MNIST_analysis = analyze_model(cfg, MNISTClient.model, f"Client 1 Round {round_num+1}", data_loaders=data_loaders, prefix="MNISTClient_")
            wandb_results, figures_to_close = log_analysis(wandb_results, MNIST_analysis, figures_to_close)

            # Analyze FashionClient.model
            Fashion_analysis = analyze_model(cfg, FashionClient.model, f"Client 2 Round {round_num+1}", data_loaders=data_loaders, prefix="FashionClient_")
            wandb_results, figures_to_close = log_analysis(wandb_results, Fashion_analysis, figures_to_close)

        # Analyze server.global_model
        server_analysis = analyze_model(cfg, server.global_model, f"Server Round {round_num+1}", data_loaders=data_loaders, prefix="server_")
        wandb_results, figures_to_close = log_analysis(wandb_results, server_analysis, figures_to_close)

        if cfg.plot_independent_latents:
            ind_fig = plot_latent_per_client(MNISTClient.model, FashionClient.model, data_loaders, title=f"Federated Round {round_num+1}", device=cfg.device)
            wandb_results.update({
                "independent_latents": wandb.Image(ind_fig, caption="Independent Latents"),
            })
            figures_to_close.append(ind_fig)

        wandb.log(wandb_results, step=round_num + 1)
        print(f"Federated Round {round_num+1}/{num_rounds}, MNIST Train Loss: {mnist_train_loss_avg:.4f}, Fashion Train Loss: {fashion_train_loss_avg:.4f}, MNIST Test Loss: {wandb_results['server_mnist_test_loss']:.4f}, Fashion Test Loss: {wandb_results['server_fashion_test_loss']:.4f}")

        # Close figures
        for fig in figures_to_close:
            plt.close(fig)
        gc.collect()
    
    return server.global_model

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Convert OmegaConf to dict for wandb
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    
    # Initialize wandb
    wandb_enabled = "online" if cfg.wandb else "offline"
    wandb.init(entity="FedRL-SNU", project=cfg.project, name=cfg.name, mode=wandb_enabled, config=wandb_config)

    # Get datasets and dataloaders
    mnist_loader, mnist_test_loader, fashion_loader, fashion_test_loader = get_dataloaders(cfg)
    dataloaders = {
        "mnist_train": mnist_loader,
        "fashion_train": fashion_loader,
        "mnist_test": mnist_test_loader,
        "fashion_test": fashion_test_loader
    }

    # Get Model
    model = get_model(cfg)

    # Pring config
    print(wandb.config)

    # Federated training
    federated_model = train_federated(cfg, dataloaders, model)

    wandb.finish()

if __name__ == "__main__":
    main()
