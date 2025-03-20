import torch
from torchvision import datasets, transforms
import wandb
from dataclasses import dataclass,asdict, field
import argparse
from collections import OrderedDict

from typing import Dict
from torch.utils.data import DataLoader
import torch.nn as nn
from copy import deepcopy

import matplotlib.pyplot as plt

from utils.losses import compute_loss
from utils.eval import analyze_model, log_analysis
from utils.data import get_dataloaders
from clients import get_client
from servers import get_server
from models import get_model
from utils.visualize import plot_latent_per_client

from tqdm import tqdm
import gc



@dataclass
class Config:
    ## General Configs  
    wandb: bool = True
    num_samples: int = 10
    save_dir: str = None
    project: str = "vae"
    name: str = "central_combined"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    ## Model Configs
    model_name: str = "vae"
    latent_dim: int = 2
    conditional: bool = False
    num_total_classes: int = 20
    #hidden_dims: list = [512, 256]

    ## Client Configs
    client_type: str = "base"    # ["base", "per_enc", "per_dec"]
    mnist_vae_mu_target: int = 0
    fashion_vae_mu_target: int = 0
    
    ## Server Configs
    server_type: str = "base"

    ## Federated Training Configs
    batch_size: int = 64
    lr: float = 1e-3
    num_rounds: int = 200
    local_epochs: int = 1

    ## Eval Configs
    eval_batch_size: int = 1000
    analyze_local_models_before_update: bool = False

    # Plot independent latents
    plot_independent_latents: bool = True


    @classmethod
    def federated_rounds200_epochs1(cls):
        return cls(name="federated_rounds200_epochs1", num_rounds=200, local_epochs=1)
    
    @classmethod
    def federated_rounds200_epochs1_cvae(cls):
        return cls(name="federated_rounds200_epochs1_cvae", num_rounds=200, local_epochs=1, conditional=True, client_type="base", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True)
    
    @classmethod
    def federated_per_enc(cls):
        return cls(name="federated_per_enc_1", client_type="per_enc", num_rounds=200, local_epochs=1, analyze_local_models_before_update=False, plot_independent_latents=True,
                   mnist_vae_mu_target=1, fashion_vae_mu_target=-1)

    @classmethod
    def federated_per_dec(cls):
        return cls(name="federated_per_dec_5", client_type="per_dec", num_rounds=200, local_epochs=1, analyze_local_models_before_update=True, plot_independent_latents=False)
    
    def asdict(self):
        return asdict(self)

def get_config(exp_name="federated_rounds200_epochs1"):
    return getattr(Config, exp_name)()

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
            # Analyzie MNISTClient.model
            MNIST_analysis = analyze_model(cfg, MNISTClient.model, f"Client 1 Round {round_num+1}", data_loaders=data_loaders, prefix="MNISTClient_")
            wandb_results, figures_to_close = log_analysis(wandb_results, MNIST_analysis, figures_to_close)
            
            # Analyzie FashionClient.model
            Fashion_analysis = analyze_model(cfg, FashionClient.model, f"Client 2 Round {round_num+1}", data_loaders=data_loaders, prefix="FashionClient_")
            wandb_results, figures_to_close = log_analysis(wandb_results, Fashion_analysis, figures_to_close)

        # Update Client Models
        MNISTClient.update_model(server.global_model.state_dict())
        FashionClient.update_model(server.global_model.state_dict())
        
        mnist_train_loss_avg = compute_loss(MNISTClient.model, mnist_loader, cfg.device, mu_target=cfg.mnist_vae_mu_target) 
        fashion_train_loss_avg = compute_loss(FashionClient.model, fashion_loader, cfg.device, mu_target=cfg.fashion_vae_mu_target)

        wandb_results.update({
            "MNIST_train_loss": mnist_train_loss_avg,
            "Fashion_train_loss": fashion_train_loss_avg,
        })
        
        if not cfg.analyze_local_models_before_update:
            # Analyzie MNISTClient.model
            MNIST_analysis = analyze_model(cfg, MNISTClient.model, f"Client 1 Round {round_num+1}", data_loaders=data_loaders, prefix="MNISTClient_")
            wandb_results, figures_to_close = log_analysis(wandb_results, MNIST_analysis, figures_to_close)

            # Analyzie FashionClient.model
            Fashion_analysis = analyze_model(cfg, FashionClient.model, f"Client 2 Round {round_num+1}", data_loaders=data_loaders, prefix="FashionClient_")
            wandb_results, figures_to_close = log_analysis(wandb_results, Fashion_analysis, figures_to_close)

        # Analyzie server.global_model
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

# Run training
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="federated_rounds200_epochs1")
    args = parser.parse_args()
    cfg = get_config(exp_name=args.name)
    wandb_enabled = "online" if cfg.wandb else "offline"
    wandb.init(entity="FedRL-SNU", project=cfg.project, name=cfg.name, mode=wandb_enabled)

    # wandb config
    wandb.config.update(cfg.asdict())

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

    # # Get Clients
    # clients = get_clients(cfg, dataloaders)

    # # get server

    # Federated training
    federated_model = train_federated(cfg, dataloaders, model)

    wandb.finish()

