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
import os



@dataclass
class Config:
    ## General Configs  
    wandb: bool = True
    num_samples: int = 10
    
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

    ## Save Configs
    save_dir: str = "/131_data/geeho/FedVAE"
    save_interval: int = 10

    # Plot independent latents
    plot_independent_latents: bool = True

    # Distance Based loss
    alpha: float = 0
    sample_p: float = 0

    @classmethod
    def centralized_rounds200_epochs1(cls):
        return cls(name="centralized_fashion_rounds200_epochs1", num_rounds=200, local_epochs=1)
    
    def asdict(self):
        return asdict(self)

def get_config(exp_name="federated_rounds200_epochs1"):
    return getattr(Config, exp_name)()

# Federated training
def train_federated(cfg, data_loaders: Dict[str, DataLoader], model: nn.Module):

    wandb_results = {}

    # Get Clients
    FashionClient = get_client(cfg, deepcopy(model), data_loaders["fashion_train"], cfg.fashion_vae_mu_target)

    num_rounds = cfg.num_rounds
    local_epochs = cfg.local_epochs
    
    for round_num in tqdm(range(num_rounds)):
        
        # Train Clients
        updated_model_state_dict, loss_dict = FashionClient.train(local_epochs)
        wandb_results.update(loss_dict, step=round_num + 1)
        
        ## Eval && analysis
        figures_to_close = []
        fashion_train_loss, fashion_train_recon_loss, fashion_train_kl_loss, fashion_train_dist_loss = compute_loss(FashionClient.model, fashion_loader, cfg.device, mu_target=cfg.fashion_vae_mu_target, alpha=cfg.alpha)
        
        wandb_results.update({
            "Fashion_train_loss": fashion_train_loss,
            "Fashion_train_recon_loss": fashion_train_recon_loss,
            "Fashion_train_kl_loss": fashion_train_kl_loss,
            "Fashion_train_dist_loss": fashion_train_dist_loss
        })
        
        if not cfg.analyze_local_models_before_update:

            # Analyzie FashionClient.model
            #FashionClient_latent_fig, FashionClient_mnist_recon_fig, FashionClient_fashion_recon_fig, FashionClient_manifold_fig, FashionClient_mnist_test_loss_avg, FashionClient_fashion_test_loss_avg = analyze_model(cfg, FashionClient.model, f"Client 2 Round {round_num+1}", data_loaders=data_loaders)
            Fashion_analysis = analyze_model(cfg, FashionClient.model, f"Client 2 Round {round_num+1}", data_loaders=data_loaders, prefix="FashionClient_")
            wandb_results, figures_to_close = log_analysis(wandb_results, Fashion_analysis, figures_to_close)

        wandb.log(wandb_results, step=round_num + 1)
        print(f"Federated Round {round_num+1}/{num_rounds}, Fashion Train Loss: {fashion_train_loss:.4f}")

        # Close figures
        for fig in figures_to_close:
            plt.close(fig)
        gc.collect()

        if round_num % cfg.save_interval == 0:
            save_path = os.path.join(cfg.save_dir, "fashion_mnist", cfg.name, f"round_{round_num}.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({"state_dict": FashionClient.model.state_dict(), "round": round_num}, save_path)

    
    #wandb.finish()
    return FashionClient.model

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
    single_vae_model = train_federated(cfg, dataloaders, model)

    wandb.finish()

