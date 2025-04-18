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
from utils.visualize import plot_latent_per_client, log_synthetic_batch_wandb

from tqdm import tqdm
import gc
from Config import Config, get_config
import copy
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Tuple, List

# Federated training
def train_federated(cfg, data_loaders: Dict[str, DataLoader], model: nn.Module):

    wandb_results = {}


    client_information = {}
    for idx, client_name in enumerate(["MNISTClient", "FashionClient"]):
        client_information[client_name] = {}
        client_information[client_name]["client_idx"] = idx
    mnist_information = deepcopy(client_information["MNISTClient"])
    fashion_information = deepcopy(client_information["FashionClient"])

    # Get Clients
    if cfg.training_type == "central":
        MNISTClient = get_client(cfg, deepcopy(model), data_loaders["combined_train"], cfg.mnist_vae_mu_target)
        FashionClient = get_client(cfg, deepcopy(model), data_loaders["combined_train"], cfg.fashion_vae_mu_target)
    else:
        #additional_information = {}
        #num_of_clients = cfg.num_clients
        # embedding_dim = cfg.embedding_dim
        # client_embeddings = torch.randn(num_of_clients, embedding_dim).to(cfg.device)
        # client_embeddings.requires_grad = False
        # client_embeddings = F.normalize(client_embeddings, dim=1)
        # additional_information["client_embeddings"] = client_embeddings

        MNISTClient = get_client(cfg, deepcopy(model), data_loaders["mnist_train"], cfg.mnist_vae_mu_target, **mnist_information)
        FashionClient = get_client(cfg, deepcopy(model), data_loaders["fashion_train"], cfg.fashion_vae_mu_target, **fashion_information)

    # Get Server
    server = get_server(cfg, deepcopy(model))

    num_rounds = cfg.num_rounds
    local_epochs = cfg.local_epochs
    
    for round_num in tqdm(range(num_rounds)):
        # Update Client Models
        MNISTClient.update_model(server.global_model.state_dict())
        FashionClient.update_model(server.global_model.state_dict())

        # Train Clients
        client_weights = [
            MNISTClient.train(local_epochs),
            FashionClient.train(local_epochs)
        ]


        
        ## Eval && analysis
        figures_to_close = []
        
        #if cfg.analyze_local_models_before_update:
        # Analyzie MNISTClient.model
        MNIST_analysis = analyze_model(cfg, MNISTClient.model, f"Client 1 Round {round_num+1}", data_loaders=data_loaders, prefix="MNISTClient_", **mnist_information, compare_model=server.global_model)
        wandb_results, figures_to_close = log_analysis(wandb_results, MNIST_analysis, figures_to_close)
        
        # Analyzie FashionClient.model
        Fashion_analysis = analyze_model(cfg, FashionClient.model, f"Client 2 Round {round_num+1}", data_loaders=data_loaders, prefix="FashionClient_", **fashion_information, compare_model=server.global_model)
        wandb_results, figures_to_close = log_analysis(wandb_results, Fashion_analysis, figures_to_close)


        
        #mnist_train_loss, mnist_train_recon_loss, mnist_train_kl_loss = compute_loss(cfg, MNISTClient.model, mnist_loader, cfg.device, mu_target=cfg.mnist_vae_mu_target) 
        MNIST_loss_dict = compute_loss(cfg, MNISTClient.model, data_loaders["mnist_train"], cfg.device, mu_target=cfg.mnist_vae_mu_target, **mnist_information)
        # mnist_train_loss = MNIST_loss_dict['total_loss']
        # mnist_train_recon_loss = MNIST_loss_dict['recon_loss']
        # mnist_train_kl_loss = MNIST_loss_dict['kl_loss']
        for key in MNIST_loss_dict:
            wandb_results[f"MNIST_train_{key}"] = MNIST_loss_dict[key]

        #fashion_train_loss, fashion_train_recon_loss, fashion_train_kl_loss = compute_loss(cfg, FashionClient.model, fashion_loader, cfg.device, mu_target=cfg.fashion_vae_mu_target)
        Fashion_loss_dict = compute_loss(cfg, FashionClient.model, data_loaders["fashion_train"], cfg.device, mu_target=cfg.fashion_vae_mu_target, **fashion_information)
        # fashion_train_loss = Fashion_loss_dict['total_loss']
        # fashion_train_recon_loss = Fashion_loss_dict['recon_loss']
        # fashion_train_kl_loss = Fashion_loss_dict['kl_loss']
        for key in Fashion_loss_dict:
            wandb_results[f"Fashion_train_{key}"] = Fashion_loss_dict[key]

        # wandb_results.update({
        #     "MNIST_train_loss": mnist_train_loss,
        #     "Fashion_train_loss": fashion_train_loss,
        #     "MNIST_train_recon_loss": mnist_train_recon_loss,
        #     "Fashion_train_recon_loss": fashion_train_recon_loss,
        #     "MNIST_train_kl_loss": mnist_train_kl_loss,
        #     "Fashion_train_kl_loss": fashion_train_kl_loss
        # })

        if cfg.use_classifier:
            mnist_train_accuracy = MNIST_loss_dict['accuracy']
            fashion_train_accuracy = Fashion_loss_dict['accuracy']
            wandb_results.update({
                "MNIST_train_accuracy": mnist_train_accuracy,
                "Fashion_train_accuracy": fashion_train_accuracy
            })
        
        # Aggregate Client Weights
        server.aggregate(client_weights)


        # if not cfg.analyze_local_models_before_update:
            # # Analyzie MNISTClient.model
            # MNIST_analysis = analyze_model(cfg, MNISTClient.model, f"Client 1 Round {round_num+1}", data_loaders=data_loaders, prefix="MNISTClient_")
            # wandb_results, figures_to_close = log_analysis(wandb_results, MNIST_analysis, figures_to_close)

            # # Analyzie FashionClient.model
            # Fashion_analysis = analyze_model(cfg, FashionClient.model, f"Client 2 Round {round_num+1}", data_loaders=data_loaders, prefix="FashionClient_")
            # wandb_results, figures_to_close = log_analysis(wandb_results, Fashion_analysis, figures_to_close)

        # Analyzie server.global_model
        server_analysis = analyze_model(cfg, server.global_model, f"Server Round {round_num+1}", data_loaders=data_loaders, prefix="server_")#, **mnist_information)
        wandb_results, figures_to_close = log_analysis(wandb_results, server_analysis, figures_to_close)

        if cfg.plot_independent_latents:
            ind_fig = plot_latent_per_client(MNISTClient.model, FashionClient.model, data_loaders, title=f"Federated Round {round_num+1}", device=cfg.device, **fashion_information)
            wandb_results.update({
                "independent_latents": wandb.Image(ind_fig, caption="Independent Latents"),
            })
            figures_to_close.append(ind_fig)

        local_synthetic_data, local_synthetic_labels = generate_synthetic_data(cfg, MNISTClient.model, FashionClient.model, global_model=None, num_samples_per_client=32, device=cfg.device)
        local_synthetic_dataloader = get_synthetic_dataloader(cfg, local_synthetic_data, local_synthetic_labels)
        #visualize data in batch of synthetic_dataloader
        local_synthetic_image = log_synthetic_batch_wandb(local_synthetic_dataloader, caption = "Local Synthetic Data Batch", num_samples=64)
        wandb_results['local_synthetic_data'] = wandb.Image(local_synthetic_image)
        figures_to_close.append(local_synthetic_image)
        #wandb_results = visualize_and_log_synthetic_data(local_synthetic_dataloader, num_samples=64, caption="Local Synthetic Data Batch", wandb_results=wandb_results)
        global_synthetic_data, global_synthetic_labels = generate_synthetic_data(cfg, MNISTClient.model, FashionClient.model, global_model=server.global_model, num_samples_per_client=32, device=cfg.device)
        global_synthetic_dataloader = get_synthetic_dataloader(cfg, global_synthetic_data, global_synthetic_labels)
        global_synthetic_image = log_synthetic_batch_wandb(global_synthetic_dataloader, caption = "Global Synthetic Data Batch", num_samples=64)
        wandb_results['global_synthetic_data'] = wandb.Image(global_synthetic_image)
        figures_to_close.append(global_synthetic_image)
        #visualize data in batch of synthetic_dataloader
        #wandb_results = visualize_and_log_synthetic_data(global_synthetic_dataloader, num_samples=64, caption="Global Synthetic Data Batch", wandb_results=wandb_results)

        wandb.log(wandb_results, step=round_num + 1)
        #print(f"Federated Round {round_num+1}/{num_rounds}, MNIST Train Loss: {mnist_train_loss:.4f}, Fashion Train Loss: {fashion_train_loss:.4f}, MNIST Test Loss: {wandb_results['server_mnist_test_total_loss']:.4f}, Fashion Test Loss: {wandb_results['server_fashion_test_total_loss']:.4f}")

        # Close figures
        for fig in figures_to_close:
            plt.close(fig)
        gc.collect()

    if cfg.save_dir is not None:
        # Save the model
        torch.save(server.global_model.state_dict(), f"{cfg.save_dir}/{cfg.name}.pth")
        print(f"Model saved to {cfg.save_dir}/{cfg.name}.pth")

    if cfg.get("finetune_last_model"):
        cfg_finalmodel = copy.deepcopy(cfg)
        cfg_finalmodel.model_name = "vae"
        unconditional_servermodel = get_model(cfg_finalmodel)
        if cfg.finetune_kl_weight >=  0:
            cfg_finalmodel.kl_weight = cfg.finetune_kl_weight

        # 1. Capture last local models (before update in the *last* round)
        # Since the loop finished, MNISTClient.model and FashionClient.model hold the state
        # after the last local training epoch and before the final update from the server.
        last_mnist_local_model = copy.deepcopy(MNISTClient.model)
        last_fashion_local_model = copy.deepcopy(FashionClient.model)

        last_mnist_local_model.eval()
        last_fashion_local_model.eval()

        # 2. Generate synthetic data
        # num_synthetic_samples_per_client = cfg.get("num_synthetic_samples_per_client", 50000)
        # synthetic_data_list = []
        # synthetic_labels_list = []
        # with torch.no_grad():
        #     # Generate from MNIST model
        #     z_mnist = torch.randn(num_synthetic_samples_per_client, cfg.latent_dim).to(cfg.device)
        #     if cfg.client_classifier:
        #         if cfg.finetune_gen_global_model:
        #             # Use the global model for decoding
        #             synthetic_mnist = server.global_model.decoder_forward(z_mnist, torch.zeros(num_synthetic_samples_per_client, dtype=torch.long).to(cfg.device)).cpu()
        #         else:
        #             synthetic_mnist = last_mnist_local_model.decoder_forward(z_mnist, torch.zeros(num_synthetic_samples_per_client, dtype=torch.long).to(cfg.device)).cpu()
        #     else:
        #         synthetic_mnist = last_mnist_local_model.decoder_forward(z_mnist).cpu()
        #     synthetic_data_list.append(synthetic_mnist)
        #     synthetic_labels_list.append(torch.zeros(num_synthetic_samples_per_client, dtype=torch.long).cpu())  # Assuming MNIST labels are 0

        #     # Generate from FashionMNIST model
        #     z_fashion = torch.randn(num_synthetic_samples_per_client, cfg.latent_dim).to(cfg.device)
        #     if cfg.client_classifier:
        #         if cfg.finetune_gen_global_model:
        #             # Use the global model for decoding
        #             synthetic_fashion = server.global_model.decoder_forward(z_fashion, torch.ones(num_synthetic_samples_per_client, dtype=torch.long).to(cfg.device)).cpu()
        #         else:
        #             synthetic_fashion = last_fashion_local_model.decoder_forward(z_fashion, torch.ones(num_synthetic_samples_per_client, dtype=torch.long).to(cfg.device)).cpu()
        #     else:
        #         synthetic_fashion = last_fashion_local_model.decoder_forward(z_fashion).cpu()
        #     synthetic_data_list.append(synthetic_fashion)
        #     synthetic_labels_list.append(torch.ones(num_synthetic_samples_per_client, dtype=torch.long).cpu())

        # synthetic_data = torch.cat(synthetic_data_list, dim=0)
        # # Clamp or normalize generated data if necessary, depending on expected input range
        # synthetic_data = torch.clamp(synthetic_data, 0, 1) # Assuming image data is [0, 1]
        # synthetic_labels = torch.cat(synthetic_labels_list, dim=0)
        synthetic_data, synthetic_labels = generate_synthetic_data(cfg_finalmodel, last_mnist_local_model, last_fashion_local_model, global_model=server.global_model if cfg.finetune_gen_global_model else None, num_samples_per_client=cfg.get("num_synthetic_samples_per_client", 50000), device=cfg.device)


        # 3. Create synthetic dataset and dataloader
        
        # synthetic_dataset = TensorDataset(synthetic_data, synthetic_labels)
        # synthetic_dataloader = DataLoader(
        #     synthetic_dataset,
        #     batch_size=cfg.get("finetune_batch_size", cfg.batch_size),
        #     shuffle=True
        # )

        synthetic_dataloader = get_synthetic_dataloader(cfg, synthetic_data, synthetic_labels)


        #visualize data in batch of synthetic_dataloader
        synthetic_image = log_synthetic_batch_wandb(synthetic_dataloader, caption = "synthetic_data", num_samples=64)
        wandb_results['synthetic_data'] = wandb.Image(synthetic_image)
        #breakpoint()
        wandb.log(wandb_results, step=num_rounds)
        plt.close(synthetic_image)


        # wandb_results = visualize_and_log_synthetic_data(synthetic_dataloader, num_samples=64, caption="Synthetic Data Batch", wandb_results=wandb_results)
        # wandb.log(wandb_results, step=num_rounds)


        # 4. Initialize Unconditional Model (already done before placeholder)
        
        #wandb.log({"synthetic_data": wandb.Image(synthetic_data[0].unsqueeze(0).cpu())})

        unconditional_servermodel.to(cfg.device)

        # 5. Fine-tune the unconditional model
        finetune_epochs = cfg.get("finetune_epochs", 100)
        finetune_lr = cfg.get("finetune_lr", cfg.lr)
        
        finetuning_client_cfg = copy.deepcopy(cfg)
        finetuning_client_cfg.client_type = "base"
        finetuning_client_cfg.local_epochs = 1
        finetuning_client_cfg.use_classifier = False
        finetuning_client_cfg.client_classifier = False
        finetuned_client= get_client(finetuning_client_cfg, deepcopy(unconditional_servermodel), synthetic_dataloader, 0)
        for epoch in range(finetune_epochs):
            wandb_results = {}
            figures_to_close = []
            finetuned_client.train(1)
            finetuned_client_mnist_analysis = analyze_model(finetuning_client_cfg, finetuned_client.model, f"Unconditional Server Model Round {epoch+1}", data_loaders=data_loaders, prefix="unconditional_servermodel_")
            wandb_results, figures_to_close = log_analysis(wandb_results, finetuned_client_mnist_analysis, figures_to_close)
            finetuned_client_fashion_analysis = analyze_model(finetuning_client_cfg, finetuned_client.model, f"Unconditional Server Model Round {epoch+1}", data_loaders=data_loaders, prefix="unconditional_servermodel_")
            wandb_results, figures_to_close = log_analysis(wandb_results, finetuned_client_fashion_analysis, figures_to_close)


            finetuned_synthetic_data, finetuned_synthetic_labels = generate_synthetic_data(cfg, None, None, global_model=finetuned_client.model, num_samples_per_client=32, device=cfg.device)
            finetuned_synthetic_dataloader = get_synthetic_dataloader(cfg, finetuned_synthetic_data, finetuned_synthetic_labels)
            finetuned_synthetic_image = log_synthetic_batch_wandb(finetuned_synthetic_dataloader, caption = "finetuned Synthetic Data Batch", num_samples=64)
            wandb_results['finetuned_synthetic_data'] = wandb.Image(finetuned_synthetic_image)




            wandb.log(wandb_results, step=epoch + 1 + num_rounds)

            for fig in figures_to_close:
                plt.close(fig)
            gc.collect()


            

        print("Finished fine-tuning.")
        # The fine-tuned model is now in unconditional_servermodel
        # Optionally save or return this model instead of server.global_model
        # For now, let's return the original federated model as per the function signature
        # If the fine-tuned model is the desired final output, change the return statement
        # return unconditional_servermodel
    
    return server.global_model


# ==============================================
# == Synthetic Data Generation Helper Function ==
# ==============================================

def generate_synthetic_data(cfg: Config,
                            mnist_model: nn.Module = None,
                            fashion_model: nn.Module = None,
                            global_model: nn.Module = None, # Optional for generation
                            num_samples_per_client: int = 50000,
                            device: torch.device = torch.device('cpu')
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates synthetic data using trained client-specific VAE decoders.

    Args:
        cfg: Configuration object.
        mnist_model: Trained VAE model specific to MNIST client.
        fashion_model: Trained VAE model specific to FashionMNIST client.
        global_model: Optional trained global VAE model (used if cfg.finetune_gen_global_model is True).
        num_samples_per_client: Number of samples to generate per client.
        device: Device to perform generation on.

    Returns:
        A tuple containing:
        - synthetic_data (torch.Tensor): Concatenated synthetic data (shape: 2*N, C, H, W).
        - synthetic_labels (torch.Tensor): Corresponding labels (0 for MNIST, 1 for Fashion).
    """
    print(f"Generating {num_samples_per_client} synthetic samples per client...")
    synthetic_data_list = []
    synthetic_labels_list = []

    if mnist_model is not None:
        mnist_model.eval()
    if fashion_model is not None:
        fashion_model.eval()
    if global_model:
        global_model.eval()

    with torch.no_grad():
        # --- Generate from MNIST model ---
        z_mnist = torch.randn(num_samples_per_client, cfg.latent_dim).to(device)
        # Determine which decoder to use
        decoder_mnist = mnist_model
        if global_model is not None:
            print("  Using global model decoder for MNIST generation.")
            decoder_mnist = global_model

        # Call the appropriate decoder forward method
        if getattr(cfg, 'client_classifier', False) or getattr(cfg, 'use_classifier', False): # Check if model expects class label
            # Assuming label 0 for MNIST for the specific decoder
            synthetic_mnist = decoder_mnist.decoder_forward(z_mnist, torch.zeros(num_samples_per_client, dtype=torch.long).to(device)).cpu()
        else:
            synthetic_mnist = decoder_mnist.decoder_forward(z_mnist).cpu()

        synthetic_data_list.append(synthetic_mnist)
        synthetic_labels_list.append(torch.zeros(num_samples_per_client, dtype=torch.long).cpu())
        print("  Generated MNIST samples.")

        # --- Generate from FashionMNIST model ---
        z_fashion = torch.randn(num_samples_per_client, cfg.latent_dim).to(device)
         # Determine which decoder to use
        decoder_fashion = fashion_model
        if global_model is not None:
            print("  Using global model decoder for FashionMNIST generation.")
            decoder_fashion = global_model

        # Call the appropriate decoder forward method
        if getattr(cfg, 'client_classifier', False) or getattr(cfg, 'use_classifier', False): # Check if model expects class label
             # Assuming label 1 for FashionMNIST for the specific decoder
            synthetic_fashion = decoder_fashion.decoder_forward(z_fashion, torch.ones(num_samples_per_client, dtype=torch.long).to(device)).cpu()
        else:
            synthetic_fashion = decoder_fashion.decoder_forward(z_fashion).cpu()

        synthetic_data_list.append(synthetic_fashion)
        synthetic_labels_list.append(torch.ones(num_samples_per_client, dtype=torch.long).cpu())
        print("  Generated FashionMNIST samples.")

    # Concatenate and potentially normalize/clamp
    synthetic_data = torch.cat(synthetic_data_list, dim=0)
    # Clamp or normalize generated data if necessary, e.g., assuming image data is [0, 1]
    synthetic_data = torch.clamp(synthetic_data, 0, 1)
    synthetic_labels = torch.cat(synthetic_labels_list, dim=0)
    print(f"Total synthetic data shape: {synthetic_data.shape}")

    return synthetic_data, synthetic_labels




def get_synthetic_dataloader(cfg, synthetic_data, synthetic_labels):
    synthetic_dataset = TensorDataset(synthetic_data, synthetic_labels)
    synthetic_dataloader = DataLoader(
        synthetic_dataset,
        batch_size=cfg.get("finetune_batch_size", cfg.batch_size),
        shuffle=True
    )
    return synthetic_dataloader

# Run training
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="federated_rounds200_epochs1")
    args = parser.parse_args()
    cfg = get_config(exp_name=args.name)

    #print exp information
    print(f"client_type: {cfg.client_type}")
    print(f"model_name: {cfg.model_name}")
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
    combined_test_dataset = ConcatDataset([mnist_test_loader.dataset, fashion_test_loader.dataset])
    combined_test_loader = DataLoader(
        combined_test_dataset,
        batch_size=cfg.eval_batch_size*2,
        shuffle=False
    )
    dataloaders = {
        "mnist_train": mnist_loader,
        "fashion_train": fashion_loader,
        "mnist_test": mnist_test_loader,
        "fashion_test": fashion_test_loader,
        "combined_train": combined_train_loader,
        "combined_test": combined_test_loader
    }

    # Get Model
    model = get_model(cfg)

    # # Get Clients
    # clients = get_clients(cfg, dataloaders)

    # # get server

    # Federated training
    federated_model = train_federated(cfg, dataloaders, model)

    wandb.finish()

