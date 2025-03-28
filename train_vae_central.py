import torch
from torchvision import datasets, transforms
import wandb
from dataclasses import dataclass, asdict
import argparse
from collections import OrderedDict
from typing import Dict
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
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
from utils.losses import vae_loss
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from utils.logging_utils import AverageMeter

@dataclass
class Config:
    # General Configs
    wandb: bool = True
    num_samples: int = 10
    save_dir: str = None
    project: str = "vae"
    name: str = "central_combined"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    training_type: str = "federated"  # "federated" or "centralized"

    # Model Configs
    model_name: str = "vae"
    latent_dim: int = 2
    batch_norm: bool = False
    conditional: bool = False
    num_total_classes: int = 20

    # Client Configs
    client_type: str = "base"
    mnist_vae_mu_target: int = 0
    fashion_vae_mu_target: int = 0
    kl_weight: float = 1

    # Server Configs
    server_type: str = "base"

    # Federated Training Configs
    batch_size: int = 64
    lr: float = 1e-3
    num_rounds: int = 200
    local_epochs: int = 1

    # Eval Configs
    eval_batch_size: int = 1000
    analyze_local_models_before_update: bool = False
    plot_independent_latents: bool = True
    manifold: bool = True
    use_classifier: bool = False
    reduction: str = 'sum'
    problabelfeatures: bool = False
    temperature: float = 1.0

    @classmethod
    def federated_rounds200_epochs1(cls):
        return cls(name="federated_rounds200_epochs1", num_rounds=200, local_epochs=1)

    @classmethod
    def centralized_rounds200(cls):
        return cls(name="centralized_rounds200", training_type="centralized", num_rounds=200, local_epochs=1)

    # Add other class methods as in your original code...
    @classmethod
    def central200_wclassifier_reductionsum(cls):
        return cls(name="central200_epochs1_wclassifier_reductionsum", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='sum', training_type="centralized")

    @classmethod
    def central200_wclassifier_reductionsum_latentdim22(cls):
        return cls(name="central200_epochs1_wclassifier_reductionsum_latentdim22", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='sum', training_type="centralized", latent_dim=22)


    def asdict(self):
        return asdict(self)

def get_config(exp_name="federated_rounds200_epochs1"):
    return getattr(Config, exp_name)()

def compute_batch_loss(cfg, model, batch, device, mu_target=0):
    x, _ = batch
    x = x.to(device)
    recon_x, mu, logvar = model(x)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + cfg.kl_weight * kl_loss
    return total_loss

def train_centralized(cfg, train_loader, model, data_loaders):
    """
    Train a VAE model centrally on a combined dataset.
    
    Args:
        cfg: Configuration object with attributes (device, num_rounds, lr, conditional, kl_weight, reduction)
        train_loader: DataLoader for the combined training dataset
        model: VAE model to train
        data_loaders: Dictionary of DataLoaders for evaluation (e.g., mnist_test, fashion_test)
    
    Returns:
        model: Trained VAE model
    """
    # Move model to device and set up optimizer
    model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    classifier_loss = nn.CrossEntropyLoss(reduction=cfg.reduction)
    # Training loop over epochs
    for epoch in tqdm(range(cfg.num_rounds)):
        model.train()
        loss_meter = AverageMeter('Loss', ':.2f')
        recon_loss_meter = AverageMeter('Recon Loss', ':.2f')
        kl_loss_meter = AverageMeter('KL Loss', ':.2f')
        classifier_loss_meter = AverageMeter('Classifier Loss', ':.2f')
        unique_values_set = set()

        # Process each batch
        for data, target in train_loader:
            data = data.to(cfg.device)
            target = target.to(cfg.device)
            optimizer.zero_grad()

            # Model forward pass (conditional or unconditional)
            if cfg.conditional:
                recon_batch, mu, log_var, z, class_output = model(data, return_classfier_output=cfg.use_classifier)
            else:
                recon_batch, mu, log_var, z, class_output = model(data, return_classfier_output=cfg.use_classifier)

            # Compute loss
            recon_loss, kl_loss = vae_loss(recon_batch, data, mu, log_var, mu_target=0, reduction=cfg.reduction)
            vae_total_loss = recon_loss + cfg.kl_weight * kl_loss

            # Classifier Loss
            cl = classifier_loss(class_output/cfg.temperature, target)

            loss = vae_total_loss + cl

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            # Update metrics
            batch_size = data.size(0)
            loss_meter.update(loss.item(), batch_size)
            recon_loss_meter.update(recon_loss.item(), batch_size)
            kl_loss_meter.update(kl_loss.item(), batch_size)
            classifier_loss_meter.update(cl.item(), batch_size)


            # Collect unique targets
            batch_unique = set(target.cpu().numpy().flatten())
            unique_values_set.update(batch_unique)

        # Log training metrics
        print(f"Epoch {epoch+1}/{cfg.num_rounds}, {loss_meter}, {recon_loss_meter}, {kl_loss_meter}, {classifier_loss_meter}")
        print(f"Unique targets: {unique_values_set}")

        # Evaluate on test sets
        model.eval()
        with torch.no_grad():
            mnist_test_loss_dict = compute_loss(cfg, model, data_loaders["mnist_test"], cfg.device, mu_target=0)
            fashion_test_loss_dict = compute_loss(cfg, model, data_loaders["fashion_test"], cfg.device, mu_target=0)
            analysis = analyze_model(cfg, model, f"Epoch {epoch+1}", data_loaders=data_loaders, prefix="centralized_")

        # Log results to wandb
        wandb_results = {
            "train_loss": loss_meter.avg,
            "train_recon_loss": recon_loss_meter.avg,
            "train_kl_loss": kl_loss_meter.avg,
            "mnist_test_loss": mnist_test_loss_dict["total_loss"],
            "fashion_test_loss": fashion_test_loss_dict["total_loss"],
            "mnist_test_recon_loss": mnist_test_loss_dict["recon_loss"],
            "fashion_test_recon_loss": fashion_test_loss_dict["recon_loss"],
            "mnist_test_kl_loss": mnist_test_loss_dict["kl_loss"],
            "fashion_test_kl_loss": fashion_test_loss_dict["kl_loss"],
            "classifier_loss": classifier_loss_meter.avg,
            "mnist_test_accuracy": mnist_test_loss_dict["accuracy"],
            "fashion_test_accuracy": fashion_test_loss_dict["accuracy"],
            # Additional metrics from analysis can be added here
        }
        wandb.log(wandb_results, step=epoch + 1)

    return model

def train_federated(cfg, data_loaders: Dict[str, DataLoader], model: nn.Module):
    # Your existing train_federated function remains unchanged...
    wandb_results = {}
    MNISTClient = get_client(cfg, deepcopy(model), data_loaders["mnist_train"], cfg.mnist_vae_mu_target)
    FashionClient = get_client(cfg, deepcopy(model), data_loaders["fashion_train"], cfg.fashion_vae_mu_target)
    server = get_server(cfg, deepcopy(model))
    num_rounds = cfg.num_rounds
    local_epochs = cfg.local_epochs

    for round_num in tqdm(range(num_rounds)):
        client_weights = [MNISTClient.train(local_epochs), FashionClient.train(local_epochs)]
        server.aggregate(client_weights)
        # ... (rest of the function as in your original code)
    return server.global_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="federated_rounds200_epochs1")
    args = parser.parse_args()
    cfg = get_config(exp_name=args.name)
    wandb_enabled = "online" if cfg.wandb else "offline"
    wandb.init(entity="FedRL-SNU", project=cfg.project, name=cfg.name, mode=wandb_enabled)

    wandb.config.update(cfg.asdict())

    mnist_loader, mnist_test_loader, fashion_loader, fashion_test_loader = get_dataloaders(cfg)
    dataloaders = {
        "mnist_train": mnist_loader,
        "fashion_train": fashion_loader,
        "mnist_test": mnist_test_loader,
        "fashion_test": fashion_test_loader,
    }

    model = get_model(cfg)

    if cfg.training_type == "federated":
        trained_model = train_federated(cfg, dataloaders, model)
    elif cfg.training_type == "centralized":
        combined_train_dataset = ConcatDataset([mnist_loader.dataset, fashion_loader.dataset])
        combined_train_loader = DataLoader(
            combined_train_dataset,
            batch_size=cfg.batch_size*2,
            shuffle=True
        )
        trained_model = train_centralized(cfg, combined_train_loader, model, dataloaders)
    else:
        raise ValueError(f"Unknown training_type: {cfg.training_type}")

    wandb.finish()