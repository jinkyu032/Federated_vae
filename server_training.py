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
from utils.visualize import plot_latent_per_client, plot_recontruction_from_noise

from tqdm import tqdm
import gc
import os


@dataclass
class Config:
    ## General Configs  
    wandb: bool = True
    num_samples: int = 10
    save_dir: str = None
    project: str = "vae"
    name: str = "server_finetuning"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    ## Model Configs
    model_name: str = "vae"
    latent_dim: int = 2
    conditional: bool = False
    num_total_classes: int = 20
    #hidden_dims: list = [512, 256]

    ## Client Configs
    client_type: str = "global"    # ["base", "per_enc", "per_dec", "global"]
    mnist_vae_mu_target: int = 0
    fashion_vae_mu_target: int = 0
    # global_vae_mu_target: int = 0
    
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

    ## Model Paths
    fashion_client_model_path: str = "/131_data/geeho/FedVAE/fashion_mnist/centralized_fashion_rounds200_epochs1/round_100.pth"
    mnist_client_model_path: str = "/131_data/geeho/FedVAE/mnist/centralized_mnist_rounds200_epochs1/round_100.pth"

    @classmethod
    def federated_rounds200_epochs1(cls):
        return cls(name="server_finetuning_rounds200_epochs1", num_rounds=200, local_epochs=1)
    
    def asdict(self):
        return asdict(self)

def get_config(exp_name="federated_rounds200_epochs1"):
    return getattr(Config, exp_name)()

# Federated training
def train_federated(cfg, data_loaders: Dict[str, DataLoader], model: nn.Module):

    wandb_results = {}

    # Get Clients
    global_client = get_client(cfg, deepcopy(model), data_loaders["mnist_train"], cfg.mnist_vae_mu_target)
    global_client.iterations = len(data_loaders["mnist_train"])

    ## Load Local Models
    mnist_client_model_state_dict = torch.load(cfg.mnist_client_model_path)['state_dict']
    fashion_client_model_state_dict = torch.load(cfg.fashion_client_model_path)['state_dict']

    global_client.get_local_models(mnist_client_model_state_dict, fashion_client_model_state_dict)
    
    num_rounds = cfg.num_rounds
    local_epochs = cfg.local_epochs

    for round_num in tqdm(range(num_rounds)):

        # Analyze local decoders
        reconstructed_mnist = plot_recontruction_from_noise(cfg, global_client.mnist_client_model, num_samples=10, device=cfg.device, mu=cfg.mnist_vae_mu_target)
        reconstructed_fashion = plot_recontruction_from_noise(cfg, global_client.fashion_client_model, num_samples=10, device=cfg.device, mu=cfg.fashion_vae_mu_target)

        wandb_results.update(
            {
                "MNISTClient_reconstructed_mnist_from_noise": wandb.Image(reconstructed_mnist),
                "FashionClient_reconstructed_fashion_from_noise": wandb.Image(reconstructed_fashion)
            }, step=round_num + 1
        )
        
        # Train Clients
        updated_model_state_dict, loss_dict = global_client.train(local_epochs)
        wandb_results.update(loss_dict, step=round_num + 1)

        ## Eval && analysis
        figures_to_close = []

        # Analyzie server.global_model
        server_analysis = analyze_model(cfg, global_client.model, f"Server Round {round_num+1}", data_loaders=data_loaders, prefix="server_")
        wandb_results, figures_to_close = log_analysis(wandb_results, server_analysis, figures_to_close)

        wandb.log(wandb_results, step=round_num + 1)
        print(f"Server Round {round_num+1}/{num_rounds}, MNIST Test Loss: {wandb_results['server_mnist_test_loss']:.4f}, Fashion Test Loss: {wandb_results['server_fashion_test_loss']:.4f}")

        # Close figures
        for fig in figures_to_close:
            plt.close(fig)
        gc.collect()
    
        # Save Model
        if round_num % cfg.save_interval == 0:
            save_path = os.path.join(cfg.save_dir, "mnist_fashion", cfg.name, f"round_{round_num}.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({"state_dict": global_client.model.state_dict(), "round": round_num}, save_path)    
    

    #wandb.finish()
    return global_client.model

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

