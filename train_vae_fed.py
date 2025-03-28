import torch
from torchvision import datasets, transforms
import wandb
from dataclasses import dataclass,asdict, field
import argparse
from collections import OrderedDict

from typing import Dict
from torch.utils.data import DataLoader, ConcatDataset
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
    batch_norm: bool = False
    conditional: bool = False
    num_total_classes: int = 20

    ## Client Configs
    client_type: str = "base"    # ["base", "per_enc", "per_dec"]
    mnist_vae_mu_target: int = 0
    fashion_vae_mu_target: int = 0
    kl_weight: float = 1
    
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
    manifold: bool = True
    use_classifier: bool = False
    reduction: str = 'sum'
    problabelfeatures: bool = False
    temperature: float = 1.0
    training_type: str = "fed"  # ["central", "fed"]
    classifier_bias: bool = True
    cosineclassifier: bool = False

    @classmethod
    def federated_rounds200_epochs1(cls):
        return cls(name="federated_rounds200_epochs1", num_rounds=200, local_epochs=1)

    @classmethod
    def federated_rounds200_epochs1_reductionmean(cls):
        return cls(name="federated_rounds200_epochs1_reductionmean", num_rounds=200, local_epochs=1, reduction='mean')
        
    @classmethod
    def federated_rounds200_epochs1_latentdim22(cls):
        return cls(name="federated_rounds200_epochs1_latentdim22", num_rounds=200, local_epochs=1, latent_dim = 22, manifold=False)

    @classmethod
    def federated_rounds200_epochs1_latentdim22_klw001(cls):
        return cls(name="federated_rounds200_epochs1_latentdim22_klw001", num_rounds=200, local_epochs=1, latent_dim = 22, manifold=False, kl_weight=0.01)
    @classmethod
    def federated_rounds200_epochs1_latentdim42(cls):
        return cls(name="federated_rounds200_epochs1_latentdim42", num_rounds=200, local_epochs=1, latent_dim = 42, manifold=False)

    @classmethod
    def federated_rounds200_epochs1_latentdim1024(cls):
        return cls(name="federated_rounds200_epochs1_latentdim1024", num_rounds=200, local_epochs=1, latent_dim = 1024, manifold=False)
    
    @classmethod
    def federated_rounds200_epochs1_cvae(cls):
        return cls(name="federated_rounds200_epochs1_cvae", num_rounds=200, local_epochs=1, conditional=True, client_type="base", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True)
        
    @classmethod
    def federated_rounds200_epochs1_cvae_latentdim22(cls):
        return cls(name="federated_rounds200_epochs1_cvae_latentdim22", num_rounds=200, local_epochs=1, conditional=True, client_type="base", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, latent_dim=22)
        

    @classmethod
    def federated_rounds200_epochs1_wclassifier_reductionmean(cls):
        return cls(name="federated_rounds200_epochs1_wclassifier_reductionmean", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='mean')

    @classmethod
    def federated_rounds200_epochs1_wclassifier_reductionmean_differentmu(cls):
        return cls(name="federated_rounds200_epochs1_wclassifier_reductionmean_differentmu", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=5, fashion_vae_mu_target=-5, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='mean')
    
    @classmethod
    def federated_rounds200_epochs1_wclassifier_reductionmean_differentmu_latentdim22(cls):
        return cls(name="federated_rounds200_epochs1_wclassifier_reductionmean_differentmu_latentdim22", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=5, fashion_vae_mu_target=-5, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='mean', latent_dim=22)
    
    @classmethod
    def federated_rounds200_epochs1_wclassifier_reductionsum_differentmu(cls):
        return cls(name="federated_rounds200_epochs1_wclassifier_reductionsum_differentmu", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=5, fashion_vae_mu_target=-5, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='sum')
    
    @classmethod
    def federated_rounds200_epochs1_wclassifier_reductionsum_differentmu_latentdim22(cls):
        return cls(name="federated_rounds200_epochs1_wclassifier_reductionsum_differentmu_latentdim22", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=5, fashion_vae_mu_target=-5, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='sum', latent_dim=22)
    
    
    @classmethod
    def federated_rounds200_epochs1_wclassifier_reductionsum(cls):
        return cls(name="federated_rounds200_epochs1_wclassifier_reductionsum", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='sum')

    @classmethod
    def central_rounds200_epochs1_wclassifier_reductionsum(cls):
        return cls(name="central_rounds200_epochs1_wclassifier_reductionsum", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='sum', training_type="central")

    @classmethod
    def central_rounds200_epochs1_wclassifier_reductionsum_latentdim22(cls):
        return cls(name="central_rounds200_epochs1_wclassifier_reductionsum_latentdim22", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='sum', latent_dim=22, training_type="central")



    @classmethod
    def federated_rounds200_epochs1_wclassifier_reductionsum_temp2(cls):
        return cls(name="federated_rounds200_epochs1_wclassifier_reductionsum_temp2", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='sum', temperature = 2)

    @classmethod
    def federated_rounds200_epochs1_wclassifier_reductionsum_temp5(cls):
        return cls(name="federated_rounds200_epochs1_wclassifier_reductionsum_temp5", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='sum', temperature = 5)

    @classmethod
    def federated_rounds200_epochs1_wclassifier_reductionsum_temp02(cls):
        return cls(name="federated_rounds200_epochs1_wclassifier_reductionsum_temp02", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='sum', temperature = 0.2)

    @classmethod
    def federated_rounds200_epochs1_wclassifier_reductionsum_temp05(cls):
        return cls(name="federated_rounds200_epochs1_wclassifier_reductionsum_temp05", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='sum', temperature = 0.5)


    @classmethod
    def federated_rounds200_epochs1_wclassifier_reductionsum_temp2_latentdim22(cls):
        return cls(name="federated_rounds200_epochs1_wclassifier_reductionsum_temp2_latentdim22", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='sum', temperature = 2, latent_dim=22)

    @classmethod
    def federated_rounds200_epochs1_wclassifier_reductionsum_temp5_latentdim22(cls):
        return cls(name="federated_rounds200_epochs1_wclassifier_reductionsum_temp5_latentdim22", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='sum', temperature = 5, latent_dim=22)

    @classmethod
    def federated_rounds200_epochs1_wclassifier_reductionsum_temp02_latentdim22(cls):
        return cls(name="federated_rounds200_epochs1_wclassifier_reductionsum_temp02_latentdim22", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='sum', temperature = 0.2, latent_dim=22)

    @classmethod
    def federated_rounds200_epochs1_wclassifier_reductionsum_temp05_latentdim22(cls):
        return cls(name="federated_rounds200_epochs1_wclassifier_reductionsum_temp05_latentdim22", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='sum', temperature = 0.5, latent_dim=22)



    @classmethod
    def federated_rounds200_epochs1_wclassifier_reductionsum_problabelfeature(cls):
        return cls(name="federated_rounds200_epochs1_wclassifier_reductionsum_problabelfeature", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='sum', problabelfeatures=True)


    @classmethod
    def federated_rounds200_epochs1_wclassifierparallel_reductionsum_problabelfeature(cls):
        return cls(name="federated_rounds200_epochs1_wclassifierparallel_reductionsum_problabelfeature", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier_parallel", reduction='sum', problabelfeatures=True)


    @classmethod
    def federated_rounds200_epochs1_wclassifierparallel_reductionsum(cls):
        return cls(name="federated_rounds200_epochs1_wclassifierparallel_reductionsum", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier_parallel", reduction='sum')

    @classmethod
    def federated_rounds200_epochs1_wclassifier_reductionsum_latentdim22(cls):
        return cls(name="federated_rounds200_epochs1_wclassifier_reductionsum_latentdim22", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='sum', latent_dim=22)

    @classmethod
    def federated_rounds200_epochs1_wclassifier_reductionsum_latentdim22_klw001(cls):
        return cls(name="federated_rounds200_epochs1_wclassifier_reductionsum_latentdim22_klw001", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='sum', latent_dim=22, kl_weight=0.01)

    @classmethod
    def federated_rounds200_epochs1_wclassifierpartial_reductionsum_latentdim22_klw001(cls):
        return cls(name="federated_rounds200_epochs1_wclassifierpartial_reductionsum_latentdim22_klw001", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier_partial", reduction='sum', latent_dim=22, kl_weight=0.01)


    @classmethod
    def federated_rounds200_epochs1_wclassifierpartialnoreuse_reductionsum_latentdim22_klw001(cls):
        return cls(name="ederated_rounds200_epochs1_wclassifierpartialnoreuse_reductionsum_latentdim22_klw001", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier_partial_noreuse", reduction='sum', latent_dim=22, kl_weight=0.01)



    @classmethod
    def federated_rounds200_epochs1_wclassifier_reductionsum_differentmu_latentdim22_klw001(cls):
        return cls(name="federated_rounds200_epochs1_wclassifier_reductionsum_differentmu_latentdim22_klw001", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=5, fashion_vae_mu_target=-5, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='sum', latent_dim=22, kl_weight=0.01)
    
    @classmethod
    def federated_rounds200_epochs1_wclassifiernobias_reductionsum_differentmu_latentdim22_klw001(cls):
        return cls(name="federated_rounds200_epochs1_wclassifiernobias_reductionsum_differentmu_latentdim22_klw001", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=5, fashion_vae_mu_target=-5, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='sum', latent_dim=22, kl_weight=0.01, classifier_bias=False)
    
    @classmethod
    def federated_rounds200_epochs1_wcosineclassifier_reductionsum_differentmu_latentdim22_klw001(cls):
        return cls(name="federated_rounds200_epochs1_wcosineclassifier_reductionsum_differentmu_latentdim22_klw001", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=5, fashion_vae_mu_target=-5, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='sum', latent_dim=22, kl_weight=0.01, classifier_bias=False, cosineclassifier=True)
    

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
    if cfg.training_type == "central":
        MNISTClient = get_client(cfg, deepcopy(model), data_loaders["combined_train"], cfg.mnist_vae_mu_target)
        FashionClient = get_client(cfg, deepcopy(model), data_loaders["combined_train"], cfg.fashion_vae_mu_target)
    else:
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
        
        #mnist_train_loss, mnist_train_recon_loss, mnist_train_kl_loss = compute_loss(cfg, MNISTClient.model, mnist_loader, cfg.device, mu_target=cfg.mnist_vae_mu_target) 
        MNIST_loss_dict = compute_loss(cfg, MNISTClient.model, data_loaders["mnist_train"], cfg.device, mu_target=cfg.mnist_vae_mu_target)
        mnist_train_loss = MNIST_loss_dict['total_loss']
        mnist_train_recon_loss = MNIST_loss_dict['recon_loss']
        mnist_train_kl_loss = MNIST_loss_dict['kl_loss']

        #fashion_train_loss, fashion_train_recon_loss, fashion_train_kl_loss = compute_loss(cfg, FashionClient.model, fashion_loader, cfg.device, mu_target=cfg.fashion_vae_mu_target)
        Fashion_loss_dict = compute_loss(cfg, FashionClient.model, data_loaders["fashion_train"], cfg.device, mu_target=cfg.fashion_vae_mu_target)
        fashion_train_loss = Fashion_loss_dict['total_loss']
        fashion_train_recon_loss = Fashion_loss_dict['recon_loss']
        fashion_train_kl_loss = Fashion_loss_dict['kl_loss']

        wandb_results.update({
            "MNIST_train_loss": mnist_train_loss,
            "Fashion_train_loss": fashion_train_loss,
            "MNIST_train_recon_loss": mnist_train_recon_loss,
            "Fashion_train_recon_loss": fashion_train_recon_loss,
            "MNIST_train_kl_loss": mnist_train_kl_loss,
            "Fashion_train_kl_loss": fashion_train_kl_loss
        })

        if cfg.use_classifier:
            mnist_train_accuracy = MNIST_loss_dict['accuracy']
            fashion_train_accuracy = Fashion_loss_dict['accuracy']
            wandb_results.update({
                "MNIST_train_accuracy": mnist_train_accuracy,
                "Fashion_train_accuracy": fashion_train_accuracy
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
        print(f"Federated Round {round_num+1}/{num_rounds}, MNIST Train Loss: {mnist_train_loss:.4f}, Fashion Train Loss: {fashion_train_loss:.4f}, MNIST Test Loss: {wandb_results['server_mnist_test_loss']:.4f}, Fashion Test Loss: {wandb_results['server_fashion_test_loss']:.4f}")

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
    combined_train_dataset = ConcatDataset([mnist_loader.dataset, fashion_loader.dataset])
    combined_train_loader = DataLoader(
        combined_train_dataset,
        batch_size=cfg.batch_size*2,
        shuffle=True
    )
    dataloaders = {
        "mnist_train": mnist_loader,
        "fashion_train": fashion_loader,
        "mnist_test": mnist_test_loader,
        "fashion_test": fashion_test_loader,
        "combined_train": combined_train_loader
    }

    # Get Model
    model = get_model(cfg)

    # # Get Clients
    # clients = get_clients(cfg, dataloaders)

    # # get server

    # Federated training
    federated_model = train_federated(cfg, dataloaders, model)

    wandb.finish()

