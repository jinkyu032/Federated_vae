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
from typing import Optional


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
    plot_independent_latents: bool = False
    manifold: bool = False
    use_classifier: bool = False
    reduction: str = 'sum'
    problabelfeatures: bool = False
    temperature: float = 1.0
    training_type: str = "fed"  # ["central", "fed"]
    classifier_bias: bool = True
    cosineclassifier: bool = False
    analyze_latent_space: bool = False
    vq: bool = False
    commitment_weight: float = 0.2
    num_embeddings: int = 32
    vq_noquantize: bool = False

    ### Iterative Training Configs
    iterative_training: bool = False
    rounds_first_stage: int = 5
    encoder_first: bool = True
    fully_conv: bool = False
    embedding_loss_weight: float = 0.1
    embedding_dim: int = 128
    num_clients: int = 2
    client_classifier: bool = False
    separate_client_encoder: bool = False
    use_consistency_loss: bool = False
    consistency_loss_weight: float = 0.1
    num_consistency_samples: int = 32
    gen_vae_loss_weight: float = 0
    gen_global_model: bool = False
    oracle: bool = False
    latent_reconloss: bool = False

    finetune_last_model: bool = False
    finetune_gen_global_model: bool = False
    finetune_kl_weight: float = -1




    #for V2
    seed: int = 42

    ## Data & Federated Splitting
    data_dir: str = "./data"
    num_workers: int = 4
    num_clients: int = 100 # Changed default
    split_mode: str = "skew" # 'iid', 'noniid-class'('skew'), 'noniid-dirichlet'
    num_subsets: int = 5 # REQUIRED for 2-stage split
    num_classes_per_client: Optional[int] = 2 # REQUIRED for noniid-class within subset
    dirichlet_alpha: Optional[float] = 0.3    # REQUIRED for noniid-dirichlet within subset
    dirichlet_min_samples: int = 10           # Optional for dirichlet
    force_resplit: bool = False               # Optional cache control
    log_client_distribution: bool = True      # Optional logging
    client_split_path: str = "./client_data" # Optional cache path

    ## Training (Federated)
    participation_rate: float = 0.05
    local_lr_decay: float = 1.0
    # mu_target needs dynamic assignment logic based on subset
    # mnist_vae_mu_target: float = 0.0 # Placeholder/Default for subset 0
    # fashion_vae_mu_target: float = 0.0 # Placeholder/Default for subset 1

    ## Server
    server_momentum: float = 0.0 # Example server param

    ## Eval & Analysis
    eval_freq: int = 1
    num_local_clients_to_analyze: int = 2
    #analyze_latent_space: bool = False
    use_mu_for_analysis: bool = False
    mi_n_neighbors: int = 5
    mi_subsample_ratio: Optional[float] = None
    log_synthetic_freq: int = 10

    ## Fine-tuning
    #num_synthetic_samples_per_client: int = 10000 # Changed default
    finetune_batch_size: int = 128
    finetune_init_from_global: bool = True
    finetune_model_name: str = "vae"
    finetune_epochs: int = 100
    finetune_lr: Optional[float] = None
    analyze_finetuned_freq: int = 5
    log_finetuned_synthetic_freq: int = 10
    save_finetuned_model: bool = True




    def get(self, key, default=None):
        return getattr(self, key, default)



    @classmethod
    def federated_rounds200_epochs1(cls):
        return cls(name="federated_rounds200_epochs1", num_rounds=200, local_epochs=1)



    @classmethod
    def federated_rounds200_epochs1_klw001(cls):
        return cls(name="federated_rounds200_epochs1_klw001", num_rounds=200, local_epochs=1, kl_weight=0.01, manifold=True)

    @classmethod
    def federated_rounds200_epochs1_klw0(cls):
        return cls(name="federated_rounds200_epochs1_klw0", num_rounds=200, local_epochs=1, kl_weight=0, manifold=True)       


    @classmethod
    def federated_rounds200_epochs1_klw3(cls):
        return cls(name="federated_rounds200_epochs1_klw3", num_rounds=200, local_epochs=1, kl_weight=3, manifold=True)      

    @classmethod
    def federated_rounds200_epochs1_klw001_latentdim22_lastfinetune(cls):
        return cls(name="federated_rounds200_epochs1_klw001_latentdim22_lastfinetune", num_rounds=200, local_epochs=1, kl_weight=0.01, latent_dim=22, finetune_last_model=True)

    @classmethod
    def federated_rounds200_epochs1_klw0_latentdim22_lastfinetune(cls):
        return cls(name="federated_rounds200_epochs1_klw0_latentdim22_lastfinetune", num_rounds=200, local_epochs=1, kl_weight=0, latent_dim=22, finetune_last_model=True)

    @classmethod
    def federated_rounds200_epochs1_klw01_latentdim22_lastfinetune(cls):
        return cls(name="federated_rounds200_epochs1_klw01_latentdim22_lastfinetune", num_rounds=200, local_epochs=1, kl_weight=0.1, latent_dim=22, finetune_last_model=True)

    @classmethod
    def federated_rounds200_epochs1_klw01_latentdim22_lastfinetuneklw1(cls):
        return cls(name="federated_rounds200_epochs1_klw01_latentdim22_lastfinetuneklw1", num_rounds=200, local_epochs=1, kl_weight=0.1, latent_dim=22, finetune_last_model=True, finetune_kl_weight=1)


    @classmethod
    def federated_rounds200_epochs1_klw01_latentdim22_lastfinetuneklw1_globalgen(cls):
        return cls(name="federated_rounds200_epochs1_klw01_latentdim22_lastfinetuneklw1_globalgen", num_rounds=200, local_epochs=1, kl_weight=0.1, latent_dim=22, finetune_last_model=True, finetune_kl_weight=1, finetune_gen_global_model=True)


    @classmethod
    def federated_rounds200_epochs1_klw3_latentdim22_lastfinetune(cls):
        return cls(name="federated_rounds200_epochs1_klw3_latentdim22_lastfinetune", num_rounds=200, local_epochs=1, kl_weight=3, latent_dim=22, finetune_last_model=True)

    @classmethod
    def central_rounds200_epochs1(cls):
        return cls(name="central_rounds200_epochs1", num_rounds=200, local_epochs=1, training_type="central")

    @classmethod
    def central_rounds200_epochs1_latentdim22(cls):
        return cls(name="central_rounds200_epochs1_latentdim22", num_rounds=200, local_epochs=1, training_type="central", latent_dim=22)


    @classmethod
    def central_rounds200_epochs1_latentdim22_lastfinetune(cls):
        return cls(name="central_rounds200_epochs1_latentdim22_lastfinetune", num_rounds=200, local_epochs=1, training_type="central", latent_dim=22, finetune_last_model=True)

    @classmethod
    def federated_rounds0_epochs1_lastfinetune(cls):
        return cls(name="federated_rounds0_epochs1_lastfinetune", num_rounds=0, local_epochs=1, finetune_last_model=True)

    
    @classmethod
    def federated_rounds1_epochs1_lastfinetune(cls):
        return cls(name="federated_rounds1_epochs1_lastfinetune", num_rounds=1, local_epochs=1, finetune_last_model=True)

    @classmethod
    def federated_rounds1_epochs10_lastfinetune(cls):
        return cls(name="federated_rounds1_epochs10_lastfinetune", num_rounds=1, local_epochs=10, finetune_last_model=True)

    

    @classmethod
    def federated_rounds1_epochs25_lastfinetune(cls):
        return cls(name="federated_rounds1_epochs25_lastfinetune", num_rounds=1, local_epochs=25, finetune_last_model=True)

    @classmethod
    def federated_rounds1_epochs100_lastfinetune(cls):
        return cls(name="federated_rounds1_epochs100_lastfinetune", num_rounds=1, local_epochs=100, finetune_last_model=True)



    @classmethod
    def federated_rounds1_epochs10_lastfinetune_latentdim22(cls):
        return cls(name="federated_rounds1_epochs10_lastfinetune_latentdim22", num_rounds=1, local_epochs=10, finetune_last_model=True, latent_dim=22)



    @classmethod
    def federated_rounds1_epochs100_lastfinetune_latentdim22(cls):
        return cls(name="federated_rounds1_epochs100_lastfinetune_latentdim22", num_rounds=1, local_epochs=100, finetune_last_model=True, latent_dim=22)


    @classmethod
    def federated_rounds200_epochs1_lastfinetune_latentdim22(cls):
        return cls(name="federated_rounds200_epochs1_lastfinetune_latentdim22", num_rounds=200, local_epochs=1, finetune_last_model=True, latent_dim=22)

    @classmethod
    def federated_rounds200_epochs1_lastfinetune_latentdim22_globalgen(cls):
        return cls(name="federated_rounds200_epochs1_lastfinetune_latentdim22_globalgen", num_rounds=200, local_epochs=1, finetune_last_model=True, latent_dim=22, finetune_gen_global_model=True)




    @classmethod
    def federated_rounds0_epochs1_lastfinetune_clientclassifierw1(cls):
        return cls(name="federated_rounds0_epochs1_lastfinetune_clientclassifierw1", num_rounds=0, local_epochs=1, finetune_last_model=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_loss_weight=1, client_type="withclientclassifier", analyze_latent_space=False, save_dir="./saved_models")


    @classmethod
    def federated_rounds1_epochs1_lastfinetune_clientclassifierw1(cls):
        return cls(name="federated_rounds1_epochs1_lastfinetune_clientclassifierw1", num_rounds=1, local_epochs=1, finetune_last_model=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_loss_weight=1, client_type="withclientclassifier", analyze_latent_space=False)

    @classmethod
    def federated_rounds1_epochs10_lastfinetune_clientclassifierw1(cls):
        return cls(name="federated_rounds1_epochs10_lastfinetune_clientclassifierw1", num_rounds=1, local_epochs=10, finetune_last_model=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_loss_weight=1, client_type="withclientclassifier", analyze_latent_space=False)
    @classmethod
    def federated_rounds1_epochs25_lastfinetune_clientclassifierw1(cls):
        return cls(name="federated_rounds1_epochs25_lastfinetune_clientclassifierw1", num_rounds=1, local_epochs=25, finetune_last_model=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_loss_weight=1, client_type="withclientclassifier", analyze_latent_space=False)
    @classmethod
    def federated_rounds1_epochs100_lastfinetune_clientclassifierw1(cls):
        return cls(name="federated_rounds1_epochs100_lastfinetune_clientclassifierw1", num_rounds=1, local_epochs=100, finetune_last_model=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_loss_weight=1, client_type="withclientclassifier", analyze_latent_space=False)


    @classmethod
    def federated_rounds200_epochs1_lastfinetune_clientclassifierw1(cls):
        return cls(name="federated_rounds200_epochs1_lastfinetune_clientclassifierw1", num_rounds=200, local_epochs=1, finetune_last_model=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_loss_weight=1, client_type="withclientclassifier", analyze_latent_space=False)

    @classmethod
    def federated_rounds200_epochs1_lastfinetune_clientclassifierw1_latentdim22(cls):
        return cls(name="federated_rounds200_epochs1_lastfinetune_clientclassifierw1_latentdim22", num_rounds=200, local_epochs=1, finetune_last_model=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_loss_weight=1, client_type="withclientclassifier", analyze_latent_space=False, save_dir="./saved_models", latent_dim=22)

    @classmethod
    def federated_rounds200_epochs1_lastfinetune_clientclassifierw01_latentdim22(cls):
        return cls(name="federated_rounds200_epochs1_lastfinetune_clientclassifierw01_latentdim22", num_rounds=200, local_epochs=1, finetune_last_model=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_loss_weight=0.1, client_type="withclientclassifier", analyze_latent_space=False, save_dir="./saved_models", latent_dim=22)



    @classmethod
    def federated_rounds200_epochs1_lastfinetune_clientclassifierw1_globalgen(cls):
        return cls(name="federated_rounds200_epochs1_lastfinetune_clientclassifierw1_globalgen", num_rounds=200, local_epochs=1, finetune_last_model=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_loss_weight=1, client_type="withclientclassifier", analyze_latent_space=False, finetune_gen_global_model=True, save_dir="./saved_models")

    @classmethod
    def federated_rounds200_epochs1_lastfinetune_clientclassifierw1_globalgen_latentdim22(cls):
        return cls(name="federated_rounds200_epochs1_lastfinetune_clientclassifierw1_globalgen_latentdim22", num_rounds=200, local_epochs=1, finetune_last_model=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_loss_weight=1, client_type="withclientclassifier", analyze_latent_space=False, finetune_gen_global_model=True, save_dir="./saved_models", latent_dim=22)

    @classmethod
    def federated_rounds200_epochs1_lastfinetune_clientclassifierw01_globalgen_latentdim22(cls):
        return cls(name="federated_rounds200_epochs1_lastfinetune_clientclassifierw01_globalgen_latentdim22", num_rounds=200, local_epochs=1, finetune_last_model=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_loss_weight=0.1, client_type="withclientclassifier", analyze_latent_space=False, finetune_gen_global_model=True, save_dir="./saved_models", latent_dim=22)




    @classmethod
    def federated_rounds200_epochs1_reductionmean(cls):
        return cls(name="federated_rounds200_epochs1_reductionmean", num_rounds=200, local_epochs=1, reduction='mean')
        
    @classmethod
    def federated_rounds200_epochs1_latentdim22(cls):
        return cls(name="federated_rounds200_epochs1_latentdim22", num_rounds=200, local_epochs=1, latent_dim = 22, manifold=False)


    @classmethod
    def fed_latentdim2_clientclassifierw01(cls):
        return cls(name="fed_latentdim2_clientclassifierw01", num_rounds=200, local_epochs=1, latent_dim = 2, manifold=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_dim=2, embedding_loss_weight=0.1, client_type="withclientclassifier", analyze_latent_space=False)


    @classmethod
    def fed_latentdim2_clientclassifierw1(cls):
        return cls(name="fed_latentdim2_clientclassifierw1", num_rounds=200, local_epochs=1, latent_dim = 2, manifold=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_dim=2, embedding_loss_weight=1, client_type="withclientclassifier", analyze_latent_space=False)
    

    @classmethod
    def fed_latentdim2_mean_le10(cls):
        return cls(name="fed_mean_le10", num_rounds=200, local_epochs=10, latent_dim = 2, manifold=True, reduction='mean')


    @classmethod
    def fed_latentdim2_clientclassifierw1_mean(cls):
        return cls(name="fed_latentdim2_clientclassifierw1_mean", num_rounds=200, local_epochs=1, latent_dim = 2, manifold=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_dim=2, embedding_loss_weight=1, client_type="withclientclassifier", analyze_latent_space=False, reduction='mean')


    @classmethod
    def fed_latentdim22_clientclassifierw1_mean(cls):
        return cls(name="fed_latentdim22_clientclassifierw1_mean", num_rounds=200, local_epochs=1, latent_dim = 22, manifold=False, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_dim=2, embedding_loss_weight=1, client_type="withclientclassifier", analyze_latent_space=False, reduction='mean', oracle=True)

    @classmethod
    def fed_latentdim22_clientclassifierw01_mean(cls):
        return cls(name="fed_latentdim22_clientclassifierw01_mean", num_rounds=200, local_epochs=1, latent_dim = 22, manifold=False, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_dim=2, embedding_loss_weight=0.1, client_type="withclientclassifier", analyze_latent_space=False, reduction='mean', oracle=True)


    @classmethod
    def fed_latentdim22_clientclassifierw1_mean_latentrecon(cls):
        return cls(name="fed_latentdim22_clientclassifierw1_mean_latentrecon", num_rounds=200, local_epochs=1, latent_dim = 22, manifold=False, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_dim=2, embedding_loss_weight=1, client_type="withclientclassifier", analyze_latent_space=False, reduction='mean', oracle=True, latent_reconloss=True)

    @classmethod
    def fed_latentdim22_clientclassifierw01_mean_latentrecon(cls):
        return cls(name="fed_latentdim22_clientclassifierw01_mean_latentrecon", num_rounds=200, local_epochs=1, latent_dim = 22, manifold=False, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_dim=2, embedding_loss_weight=0.1, client_type="withclientclassifier", analyze_latent_space=False, reduction='mean', oracle=True, latent_reconloss=True)

    



    @classmethod
    def fed_latentdim2_clientclassifierw1_mean_le10(cls):
        return cls(name="fed_latentdim2_clientclassifierw1_mean_le10", num_rounds=200, local_epochs=10, latent_dim = 2, manifold=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_dim=2, embedding_loss_weight=1, client_type="withclientclassifier", analyze_latent_space=False, reduction='mean')


    @classmethod
    def fed_latentdim2_clientclassifierw1_mean_constreg1(cls):
        return cls(name="fed_latentdim2_clientclassifierw1_mean_constreg1", num_rounds=200, local_epochs=1, latent_dim = 2, manifold=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_dim=2, embedding_loss_weight=1, client_type="withclientclassifier", analyze_latent_space=False, reduction='mean', use_consistency_loss=True, consistency_loss_weight=1, num_consistency_samples=32)

    @classmethod
    def fed_latentdim2_clientclassifierw1_mean_constreg01(cls):
        return cls(name="fed_latentdim2_clientclassifierw1_mean_constreg01", num_rounds=200, local_epochs=1, latent_dim = 2, manifold=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_dim=2, embedding_loss_weight=1, client_type="withclientclassifier", analyze_latent_space=False, reduction='mean', use_consistency_loss=True, consistency_loss_weight=0.1, num_consistency_samples=32)




    @classmethod
    def fed_latentdim2_clientclassifierw1_mean_constreg01_globalgen(cls):
        return cls(name="fed_latentdim2_clientclassifierw1_mean_constreg01_globalgen", num_rounds=200, local_epochs=1, latent_dim = 2, manifold=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_dim=2, embedding_loss_weight=1, client_type="withclientclassifier", analyze_latent_space=False, reduction='mean', use_consistency_loss=True, consistency_loss_weight=0.1, num_consistency_samples=32, gen_global_model=True)


    @classmethod
    def fed_latentdim2_clientclassifierw1_mean_constreg01_globalgen_le10(cls):
        return cls(name="fed_latentdim2_clientclassifierw1_mean_constreg01_globalgen_le10", num_rounds=200, local_epochs=10, latent_dim = 2, manifold=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_dim=2, embedding_loss_weight=1, client_type="withclientclassifier", analyze_latent_space=False, reduction='mean', use_consistency_loss=True, consistency_loss_weight=0.1, num_consistency_samples=32, gen_global_model=True)


    @classmethod
    def fed_latentdim2_clientclassifierw1_mean_constreg01recon1_globalgen(cls):
        return cls(name="fed_latentdim2_clientclassifierw1_mean_constreg01recon1_globalgen", num_rounds=200, local_epochs=1, latent_dim = 2, manifold=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_dim=2, embedding_loss_weight=1, client_type="withclientclassifier", analyze_latent_space=False, reduction='mean', use_consistency_loss=True, consistency_loss_weight=0.1, num_consistency_samples=32, gen_global_model=True, gen_vae_loss_weight=1)


    @classmethod
    def fed_latentdim2_clientclassifierw1_mean_constreg01recon1_globalgen_le10(cls):
        return cls(name="fed_latentdim2_clientclassifierw1_mean_constreg01recon1_globalgen_le10", num_rounds=200, local_epochs=10, latent_dim = 2, manifold=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_dim=2, embedding_loss_weight=1, client_type="withclientclassifier", analyze_latent_space=False, reduction='mean', use_consistency_loss=True, consistency_loss_weight=0.1, num_consistency_samples=32, gen_global_model=True, gen_vae_loss_weight=1)


    @classmethod
    def fed_latentdim2_clientclassifierw1_mean_constreg01recon1_localgen_le10(cls):
        return cls(name="fed_latentdim2_clientclassifierw1_mean_constreg01recon1_localgen_le10", num_rounds=200, local_epochs=10, latent_dim = 2, manifold=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_dim=2, embedding_loss_weight=1, client_type="withclientclassifier", analyze_latent_space=False, reduction='mean', use_consistency_loss=True, consistency_loss_weight=0.1, num_consistency_samples=32, gen_global_model=False, gen_vae_loss_weight=1)


    @classmethod
    def fed_latentdim2_clientclassifierw1_mean_constreg01recon1_localgen(cls):
        return cls(name="fed_latentdim2_clientclassifierw1_mean_constreg01recon1_localgen", num_rounds=200, local_epochs=1, latent_dim = 2, manifold=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_dim=2, embedding_loss_weight=1, client_type="withclientclassifier", analyze_latent_space=False, reduction='mean', use_consistency_loss=True, consistency_loss_weight=0.1, num_consistency_samples=32, gen_global_model=False, gen_vae_loss_weight=1)



    @classmethod
    def fed_latentdim2_clientclassifierw1_mean_constreg0recon1_globalgen(cls):
        return cls(name="fed_latentdim2_clientclassifierw1_mean_constreg0recon1_globalgen", num_rounds=200, local_epochs=1, latent_dim = 2, manifold=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_dim=2, embedding_loss_weight=1, client_type="withclientclassifier", analyze_latent_space=False, reduction='mean', use_consistency_loss=True, consistency_loss_weight=0, num_consistency_samples=32, gen_global_model=True, gen_vae_loss_weight=1)

    @classmethod
    def fed_latentdim2_clientclassifierw1_mean_constreg0recon1_localgen(cls):
        return cls(name="fed_latentdim2_clientclassifierw1_mean_constreg0recon1_localgen", num_rounds=200, local_epochs=1, latent_dim = 2, manifold=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_dim=2, embedding_loss_weight=1, client_type="withclientclassifier", analyze_latent_space=False, reduction='mean', use_consistency_loss=True, consistency_loss_weight=0, num_consistency_samples=32, gen_global_model=False, gen_vae_loss_weight=1)






    @classmethod
    def fed_latentdim2_clientclassifierw01_mean_constregw1(cls):
        return cls(name="fed_latentdim2_clientclassifierw01_mean_constregw1", num_rounds=200, local_epochs=1, latent_dim = 2, manifold=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_dim=2, embedding_loss_weight=0.1, client_type="withclientclassifier", analyze_latent_space=False, reduction='mean', use_consistency_loss=True, consistency_loss_weight=1, num_consistency_samples=32)

    @classmethod
    def fed_latentdim2_clientclassifierw01_mean_constregw01(cls):
        return cls(name="fed_latentdim2_clientclassifierw01_mean_constregw01", num_rounds=200, local_epochs=1, latent_dim = 2, manifold=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_dim=2, embedding_loss_weight=0.1, client_type="withclientclassifier", analyze_latent_space=False, reduction='mean', use_consistency_loss=True, consistency_loss_weight=0.1, num_consistency_samples=32)

    @classmethod
    def fed_latentdim2_clientclassifierw01_mean_constregw001(cls):
        return cls(name="fed_latentdim2_clientclassifierw01_mean_constregw001", num_rounds=200, local_epochs=1, latent_dim = 2, manifold=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_dim=2, embedding_loss_weight=0.1, client_type="withclientclassifier", analyze_latent_space=False, reduction='mean', use_consistency_loss=True, consistency_loss_weight=0.01, num_consistency_samples=32)



    @classmethod
    def fed_latentdim2_clientclassifierw1_mean_constregwreconl(cls):
        return cls(name="fed_latentdim2_clientclassifierw1_mean_constregwreconl", num_rounds=200, local_epochs=1, latent_dim = 2, manifold=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_dim=2, embedding_loss_weight=1, client_type="withclientclassifier", analyze_latent_space=False, reduction='mean', use_consistency_loss=True, consistency_loss_weight=1, num_consistency_samples=32, gen_vae_loss_weight=1)






    @classmethod
    def fed_latentdim2_clientclassifierw1_mean_sepcla(cls):
        return cls(name="fed_latentdim2_clientclassifierw1_mean_sepcla", num_rounds=200, local_epochs=1, latent_dim = 2, manifold=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_dim=2, embedding_loss_weight=1, client_type="withclientclassifier", analyze_latent_space=False, reduction='mean', separate_client_encoder=True)


    @classmethod
    def fed_latentdim2_clientclassifierw01_mean(cls):
        return cls(name="fed_latentdim2_clientclassifierw01_mean", num_rounds=200, local_epochs=1, latent_dim = 2, manifold=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_dim=2, embedding_loss_weight=0.1, client_type="withclientclassifier", analyze_latent_space=False, reduction='mean')



    @classmethod
    def fed_latentdim2_clientclassifierw01_mean_sepcla(cls):
        return cls(name="fed_latentdim2_clientclassifierw01_mean_sepcla", num_rounds=200, local_epochs=1, latent_dim = 2, manifold=True, client_classifier=True, num_clients=2, model_name="vaewithclientclassifier", use_classifier=True, embedding_dim=2, embedding_loss_weight=0.1, client_type="withclientclassifier", analyze_latent_space=False, reduction='mean', separate_client_encoder=True)



    @classmethod
    def central_epochs1_convvae_fullyconv(cls):
        return cls(name="central_epochs1_convvae_fullyconv", num_rounds=200, local_epochs=1, latent_dim = 22, manifold=False, model_name="convvae", fully_conv=True, training_type="central")
    @classmethod
    def fed_epochs1_convvae_fullyconv(cls):
        return cls(name="fed_epochs1_convvae_fullyconv", num_rounds=200, local_epochs=1, latent_dim = 22, manifold=False, model_name="convvae", fully_conv=True)
    @classmethod
    def fed_epochs10_convvae_fullyconv(cls):
        return cls(name="fed_epochs10_convvae_fullyconv", num_rounds=200, local_epochs=10, latent_dim = 22, manifold=False, model_name="convvae", fully_conv=True)

    @classmethod
    def central_epochs1_convvae_fullyconv_klw001(cls):
        return cls(name="central_epochs1_convvae_fullyconv_klw001", num_rounds=200, local_epochs=1, latent_dim = 22, manifold=False, model_name="convvae", fully_conv=True, training_type="central", kl_weight=0.01)
    @classmethod
    def fed_epochs1_convvae_fullyconv_klw001(cls):
        return cls(name="fed_epochs1_convvae_fullyconv_klw001", num_rounds=200, local_epochs=1, latent_dim = 22, manifold=False, model_name="convvae", fully_conv=True, kl_weight=0.01)
    @classmethod
    def fed_epochs10_convvae_fullyconv_klw001(cls):
        return cls(name="fed_epochs10_convvae_fullyconv_klw001", num_rounds=200, local_epochs=10, latent_dim = 22, manifold=False, model_name="convvae", fully_conv=True,  kl_weight=0.01)


    @classmethod
    def central_epochs1_convvae_nofullyconv(cls):
        return cls(name="central_epochs1_convvae_nofullyconv", num_rounds=200, local_epochs=1, latent_dim = 22, manifold=False, model_name="convvae", fully_conv=False, training_type="central")
    @classmethod
    def fed_epochs1_convvae_nofullyconv(cls):
        return cls(name="fed_epochs1_convvae_nofullyconv", num_rounds=200, local_epochs=1, latent_dim = 22, manifold=False, model_name="convvae", fully_conv=False)
    @classmethod
    def fed_epochs10_convvae_nofullyconv(cls):
        return cls(name="fed_epochs10_convvae_nofullyconv", num_rounds=200, local_epochs=10, latent_dim = 22, manifold=False, model_name="convvae", fully_conv=False)

    @classmethod
    def central_epochs1_convvae_nofullyconv_klw001(cls):
        return cls(name="central_epochs1_convvae_nofullyconv_klw001", num_rounds=200, local_epochs=1, latent_dim = 22, manifold=False, model_name="convvae", fully_conv=False, training_type="central", kl_weight=0.01)
    @classmethod
    def fed_epochs1_convvae_nofullyconv_klw001(cls):
        return cls(name="fed_epochs1_convvae_nofullyconv_klw001", num_rounds=200, local_epochs=1, latent_dim = 22, manifold=False, model_name="convvae", fully_conv=False, kl_weight=0.01)
    @classmethod
    def fed_epochs10_convvae_nofullyconv_klw001(cls):
        return cls(name="fed_epochs10_convvae_nofullyconv_klw001", num_rounds=200, local_epochs=10, latent_dim = 22, manifold=False, model_name="convvae", fully_conv=False,  kl_weight=0.01)

    @classmethod
    def federated_rounds200_epochs1_numembed32_vq(cls):
        return cls(name="federated_rounds200_epochs1_numembed32_vq", num_rounds=200, local_epochs=1, latent_dim = 22, manifold=False, vq=True, model_name="vqvae", num_embeddings = 32)
    @classmethod
    def federated_rounds200_epochs10_numembed32_vq(cls):
        return cls(name="federated_rounds200_epochs10_numembed32_vq", num_rounds=200, local_epochs=10, latent_dim = 22, manifold=False, vq=True, model_name="vqvae", num_embeddings = 32)


    @classmethod
    def central_rounds200_epochs1_numembed32_vq(cls):
        return cls(name="central_rounds200_epochs1_numembed32_vq", num_rounds=200, local_epochs=1, latent_dim = 22, manifold=False, vq=True, model_name="vqvae", training_type="central", num_embeddings = 32)


    @classmethod
    def central_rounds200_epochs1_numembed32_vq_noquantize(cls):
        return cls(name="central_rounds200_epochs1_numembed32_vq_noquantize", num_rounds=200, local_epochs=1, latent_dim = 22, manifold=False, vq=True, model_name="vqvae", training_type="central", vq_noquantize=True, num_embeddings = 32)

    @classmethod
    def federated_rounds200_epochs1_numembed32_vq_noquantize(cls):
        return cls(name="federated_rounds200_epochs1_numembed32_vq_noquantize", num_rounds=200, local_epochs=1, latent_dim = 22, manifold=False, vq=True, model_name="vqvae", vq_noquantize=True, num_embeddings = 32)
    @classmethod
    def federated_rounds200_epochs10_numembed32_vq_noquantize(cls):
        return cls(name="federated_rounds200_epochs10_numembed32_vq_noquantize", num_rounds=200, local_epochs=10, latent_dim = 22, manifold=False, vq=True, model_name="vqvae", vq_noquantize=True, num_embeddings = 32)
    
    @classmethod
    def federated_rounds200_epochs10_latentdim22(cls):
        return cls(name="federated_rounds200_epochs10_latentdim22", num_rounds=200, local_epochs=10, latent_dim = 22, manifold=False)
    
    @classmethod
    def federated_rounds200_epochs10_latentdim22_iterative_ef(cls):
        return cls(name="federated_rounds200_epochs10_latentdim22_iter_ef", num_rounds=200, local_epochs=10, latent_dim = 22, manifold=False, iterative_training=True, rounds_first_stage=5, encoder_first=True)
    
    @classmethod
    def federated_rounds200_epochs10_latentdim22_iterative_df(cls):
        return cls(name="federated_rounds200_epochs10_latentdim22_iter_df", num_rounds=200, local_epochs=10, latent_dim = 22, manifold=False, iterative_training=True, rounds_first_stage=5, encoder_first=False)

    @classmethod
    def federated_rounds200_epochs1_latentdim22_klw001(cls):
        return cls(name="federated_rounds200_epochs1_latentdim22_klw001", num_rounds=200, local_epochs=1, latent_dim = 22, manifold=False, kl_weight=0.01)

    @classmethod
    def federated_rounds200_epochs10_latentdim22_klw001(cls):
        return cls(name="federated_rounds200_epochs10_latentdim22_klw001", num_rounds=200, local_epochs=10, latent_dim = 22, manifold=False, kl_weight=0.01)


    @classmethod
    def federated_rounds200_epochs1_latentdim44_klw001(cls):
        return cls(name="federated_rounds200_epochs1_latentdim44_klw001", num_rounds=200, local_epochs=1, latent_dim = 44, manifold=False, kl_weight=0.01)

    @classmethod
    def federated_rounds200_epochs1_latentdim44_klw1(cls):
        return cls(name="federated_rounds200_epochs1_latentdim44_klw1", num_rounds=200, local_epochs=1, latent_dim = 44, manifold=False, kl_weight=1)
    @classmethod
    def federated_rounds200_epochs1_latentdim44_klw3(cls):
        return cls(name="federated_rounds200_epochs1_latentdim44_klw3", num_rounds=200, local_epochs=1, latent_dim = 44, manifold=False, kl_weight=3)
    @classmethod
    def federated_rounds200_epochs1_latentdim44_klw5(cls):
        return cls(name="federated_rounds200_epochs1_latentdim44_klw5", num_rounds=200, local_epochs=1, latent_dim = 44, manifold=False, kl_weight=5)

    @classmethod
    def federated_rounds200_epochs1_latentdim22_klw5(cls):
        return cls(name="federated_rounds200_epochs1_latentdim22_klw5", num_rounds=200, local_epochs=1, latent_dim = 22, manifold=False, kl_weight=5)
    
    @classmethod
    def federated_rounds200_epochs10_latentdim22_klw5(cls):
        return cls(name="federated_rounds200_epochs10_latentdim22_klw5", num_rounds=200, local_epochs=10, latent_dim = 22, manifold=False, kl_weight=5)
    
    @classmethod
    def federated_rounds200_epochs1_latentdim22_klw3(cls):
        return cls(name="federated_rounds200_epochs1_latentdim22_klw3", num_rounds=200, local_epochs=1, latent_dim = 22, manifold=False, kl_weight=3)

    @classmethod
    def federated_rounds200_epochs10_latentdim22_klw3(cls):
        return cls(name="federated_rounds200_epochs10_latentdim22_klw3", num_rounds=200, local_epochs=10, latent_dim = 22, manifold=False, kl_weight=3)

    @classmethod
    def federated_rounds200_epochs1_latentdim22_klw0(cls):
        return cls(name="federated_rounds200_epochs1_latentdim22_klw0", num_rounds=200, local_epochs=1, latent_dim = 22, manifold=False, kl_weight=0)

    @classmethod
    def federated_rounds200_epochs1_latentdim22_klw1(cls):
        return cls(name="federated_rounds200_epochs1_latentdim22_klw1", num_rounds=200, local_epochs=1, latent_dim = 22, manifold=False, kl_weight=1)

    @classmethod
    def federated_rounds200_epochs1_latentdim22_differentmu_klw1(cls):
        return cls(name="federated_rounds200_epochs1_latentdim22_differentmu_klw1", num_rounds=200, local_epochs=1, latent_dim = 22, manifold=False, kl_weight=1, mnist_vae_mu_target=5, fashion_vae_mu_target=-5)


    @classmethod
    def federated_rounds200_epochs1_latentdim2_differentmu_klw1(cls):
        return cls(name="federated_rounds200_epochs1_latentdim2_differentmu_klw1", num_rounds=200, local_epochs=1, latent_dim = 2, manifold=False, kl_weight=1, mnist_vae_mu_target=5, fashion_vae_mu_target=-5)

    
    
    @classmethod
    def federated_rounds200_epochs1_latentdim22_differentmu_klw001(cls):
        return cls(name="federated_rounds200_epochs1_latentdim22_differentmu_klw001", num_rounds=200, local_epochs=1, latent_dim = 22, manifold=False, kl_weight=0.01, mnist_vae_mu_target=5, fashion_vae_mu_target=-5)

    


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
    def federated_rounds200_epochs1_cvae_latentdim22_klw001(cls):
        return cls(name="federated_rounds200_epochs1_cvae_latentdim22_klw001", num_rounds=200, local_epochs=1, conditional=True, client_type="base", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, latent_dim=22, kl_weight=0.01)
       

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
    def federated_rounds200_epochs1_wclassifier_reductionsum_latentdim22_klw1(cls):
        return cls(name="federated_rounds200_epochs1_wclassifier_reductionsum_latentdim22_klw1", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='sum', latent_dim=22, kl_weight=1)

    @classmethod
    def federated_rounds200_epochs1_wclassifier_reductionsum_latentdim22_klw001(cls):
        return cls(name="federated_rounds200_epochs1_wclassifier_reductionsum_latentdim22_klw001", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='sum', latent_dim=22, kl_weight=0.01)




    @classmethod
    def central_rounds200_epochs1_wclassifier_reductionsum(cls):
        return cls(name="central_rounds200_epochs1_wclassifier_reductionsum", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='sum', training_type="central")

    @classmethod
    def central_rounds200_epochs1_wclassifier_reductionsum_latentdim22(cls):
        return cls(name="central_rounds200_epochs1_wclassifier_reductionsum_latentdim22", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='sum', latent_dim=22, training_type="central")

    @classmethod
    def central_rounds200_epochs1_wclassifier_reductionsum_latentdim22_klw001(cls):
        return cls(name="central_rounds200_epochs1_wclassifier_reductionsum_latentdim22_klw001", num_rounds=200, local_epochs=1, conditional=True, client_type="withclassifier", mnist_vae_mu_target=0, fashion_vae_mu_target=0, analyze_local_models_before_update=True, use_classifier=True, model_name="vaewithclassifier", reduction='sum', latent_dim=22, training_type="central", kl_weight=0.01)



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

