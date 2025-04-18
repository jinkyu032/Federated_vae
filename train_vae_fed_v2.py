import torch
from torchvision import datasets, transforms
import wandb
from dataclasses import dataclass,asdict, field
import argparse
from collections import OrderedDict, defaultdict # Added defaultdict

from typing import Dict, List, Tuple # Added List
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset # Added TensorDataset
import torch.nn as nn
from copy import deepcopy
import numpy as np # Added numpy

import matplotlib.pyplot as plt

# Assuming these utils exist and function correctly
from utils.losses import compute_loss
from utils.eval import analyze_model, log_analysis
# Import the NEW federated data loading function
from utils.data import get_dataloaders_federated, DatasetSplit # Import the specific function
from clients import get_client # Keep using the factory pattern
from servers import get_server # Keep using the factory pattern
from models import get_model
from utils.visualize import plot_latent_per_client, log_synthetic_batch_wandb # Keep visualization

from tqdm import tqdm
import gc
from Config_v2 import Config, get_config
import copy
import time # Added time for potential timing logs
from typing import Optional # Added Optional for type hinting


# ==============================================
# == Synthetic Data Generation Helper Function ==
# ==============================================
# Keep the refactored generate_synthetic_data from previous answers
# def generate_synthetic_data(cfg: Config, client_models: List[nn.Module],
#                             global_model: nn.Module = None,
#                             num_samples_per_client: int = 50000,
#                             device: torch.device = torch.device('cpu')
#                            ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Generates synthetic data using trained client-specific VAE decoders.
#     Generalized for a list of client models. It currently assumes the first
#     half of clients are 'MNIST-like' (target 0) and the second half
#     are 'Fashion-like' (target 1). This needs adjustment for >2 subsets.
#     """
#     print(f"Generating {num_samples_per_client} synthetic samples per client type...")
#     synthetic_data_list = []
#     synthetic_labels_list = []

#     # !!! IMPORTANT ASSUMPTION for current implementation:
#     # Assumes clients are grouped by subset, and we generate based on the *first*
#     # client model found for each subset type (MNIST-like vs Fashion-like).
#     # This needs refinement if generation per specific client or subset is needed.
#     model_subset_0 = None
#     model_subset_1 = None
#     if client_models: # Check if list is not empty
#         # Crude way to get representative models - assumes ordering or grouping
#         model_subset_0 = client_models[0]
#         if len(client_models) > cfg.num_clients // cfg.num_subsets: # If there's likely a second group
#              model_subset_1 = client_models[cfg.num_clients // cfg.num_subsets]
#         else: # Only one group? Generate only one type? Or use the same model?
#              print("Warning: Only one client model group detected for generation.")
#              model_subset_1 = client_models[0] # Use the first model for both for now

#     if model_subset_0 is not None: model_subset_0.eval()
#     if model_subset_1 is not None: model_subset_1.eval()
#     if global_model: global_model.eval()

#     with torch.no_grad():
#         # Generate for Subset 0 (e.g., MNIST)
#         # if model_subset_0 is not None:
#         z_subset_0 = torch.randn(num_samples_per_client, cfg.latent_dim).to(device)
#         decoder_0 = model_subset_0
#         if global_model is not None:
#             print("  Using global model decoder for Subset 0 generation.")
#             decoder_0 = global_model

#         if getattr(cfg, 'client_classifier', False) or getattr(cfg, 'use_classifier', False):
#             # Assuming label 0 for the first subset
#             synthetic_0 = decoder_0.decoder_forward(z_subset_0, torch.zeros(num_samples_per_client, dtype=torch.long).to(device)).cpu()
#         else:
#             synthetic_0 = decoder_0.decoder_forward(z_subset_0).cpu()

#         synthetic_data_list.append(synthetic_0)
#         synthetic_labels_list.append(torch.zeros(num_samples_per_client, dtype=torch.long).cpu())
#         print("  Generated Subset 0 samples.")

#         # Generate for Subset 1 (e.g., FashionMNIST)
#         #if model_subset_1 is not None:
#         z_subset_1 = torch.randn(num_samples_per_client, cfg.latent_dim).to(device)
#         decoder_1 = model_subset_1
#         if global_model is not None:
#             print("  Using global model decoder for Subset 1 generation.")
#             decoder_1 = global_model

#         if getattr(cfg, 'client_classifier', False) or getattr(cfg, 'use_classifier', False):
#             # Assuming label 10 for the second subset (needs generalization)
#             # This needs a better way to map subset_id to a representative label
#             label_to_use = 10 # Hardcoded assumption
#             synthetic_1 = decoder_1.decoder_forward(z_subset_1, torch.full((num_samples_per_client,), label_to_use, dtype=torch.long).to(device)).cpu()
#         else:
#             label_to_use = -1 # No specific label for generation
#             synthetic_1 = decoder_1.decoder_forward(z_subset_1).cpu()

#         synthetic_data_list.append(synthetic_1)
#         # Labels should also correspond to the subset being generated from
#         synthetic_labels_list.append(torch.full((num_samples_per_client,), label_to_use, dtype=torch.long).cpu())
#         print("  Generated Subset 1 samples.")

#     if not synthetic_data_list:
#         print("Error: No synthetic data generated.")
#         return torch.empty(0), torch.empty(0)

#     synthetic_data = torch.cat(synthetic_data_list, dim=0)
#     synthetic_data = torch.clamp(synthetic_data, 0, 1)
#     synthetic_labels = torch.cat(synthetic_labels_list, dim=0)
#     print(f"Total synthetic data shape: {synthetic_data.shape}")

#     return synthetic_data, synthetic_labels

def generate_synthetic_data(cfg: Config,
                            client_models: List[nn.Module], # List of ALL trained client models
                            client_subset_assignment: Dict[int, int], # Map client_id -> subset_id
                            global_model: Optional[nn.Module] = None, # Option to use global decoder
                            num_samples_total: int = 100000, # Total samples to generate across all clients
                            device: torch.device = torch.device('cpu'),
                            gen_from_global: bool = False, # Flag to indicate if using global model,
                            client_idxs: Optional[List[int]] = None # Optional list of client indices to generate from
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates synthetic data by sampling from EACH client's trained VAE decoder.

    Args:
        cfg: Configuration object.
        client_models: List of trained client VAE models (index matches client_id).
        client_subset_assignment: Dictionary mapping client_id to subset_id.
        global_model: Optional global model. If cfg.finetune_gen_global_model is True,
                      its decoder is used instead of client decoders.
        num_samples_total: Total number of synthetic samples to generate across all clients.
        device: Device to perform generation on.

    Returns:
        A tuple containing:
        - synthetic_data (torch.Tensor): Concatenated synthetic data from all clients.
        - synthetic_labels (torch.Tensor): Corresponding representative original labels based on client's subset.
    """
    num_clients = len(client_models)
    if num_clients == 0:
        print("Warning: No client models provided for synthetic data generation.")


    # Determine samples per client
    samples_per_client = num_samples_total // num_clients
    remainder_samples = num_samples_total % num_clients
    print(f"Generating approx {samples_per_client} (+1 for first {remainder_samples}) samples per client (Total: {num_samples_total})...")

    synthetic_data_list = []
    synthetic_labels_list = []

    use_global = global_model is not None and gen_from_global
    if use_global:
        print("  Using GLOBAL model decoder for all generation.")
        global_model.eval()
    else:
        # Set all client models to eval mode
        for model in client_models: model.eval()

    # Needed for label assignment
    num_classes_total = getattr(cfg, 'num_total_classes', 20)
    num_subsets = getattr(cfg, 'num_subsets', 1) # Default to 1 subset if not specified
    classes_per_subset = int(np.ceil(num_classes_total / num_subsets))


    with torch.no_grad():
        for idx,client_model in enumerate(client_models):
            if client_idxs is not None:
                client_idx = client_idxs[idx]
            else:
                client_idx = idx
            # Determine number of samples for this specific client
            num_samples = samples_per_client + (1 if client_idx < remainder_samples else 0)
            if num_samples == 0: continue # Skip if no samples assigned

            # Choose the decoder to use
            decoder_model = global_model if use_global else client_model

            # Generate latent codes
            z_client = torch.randn(num_samples, cfg.latent_dim).to(device)

            # Determine a representative label for this client based on its subset
            if cfg.condition_subset:
                subset_id = client_subset_assignment.get(client_idx, -1) # Get subset ID
                if subset_id == -1:
                    print(f"Warning: Cannot find subset assignment for client {client_idx}. Assigning label -1.")
                    representative_label = -1
                else:
                    representative_label = subset_id
            else:
                # If not conditioning on subset, use the client index as the label
                representative_label = client_idx % num_classes_total


            # Generate samples using the chosen decoder
            if getattr(cfg, 'client_classifier', False) or getattr(cfg, 'use_classifier', False):
                # Pass the representative label if model is conditional
                synth_client = decoder_model.decoder_forward(
                    z_client,
                    torch.full((num_samples,), representative_label, dtype=torch.long).to(device)
                ).cpu()
            else:
                # Unconditional generation
                synth_client = decoder_model.decoder_forward(z_client).cpu()

            synthetic_data_list.append(synth_client)
            synthetic_labels_list.append(
                torch.full((num_samples,), representative_label, dtype=torch.long).cpu()
            )
            # Optional: Add progress print
            # if (client_idx + 1) % 10 == 0: print(f"   Generated for client {client_idx+1}/{num_clients}")


    # --- Combine and Finalize ---
    if not synthetic_data_list:
        print("Error: No synthetic data was generated.")
        return torch.empty(0), torch.empty(0)

    synthetic_data = torch.cat(synthetic_data_list, dim=0)
    synthetic_data = torch.clamp(synthetic_data, 0, 1) # Assuming image data [0, 1]
    synthetic_labels = torch.cat(synthetic_labels_list, dim=0)
    print(f"Total synthetic data generated shape: {synthetic_data.shape}")
    # Verify total samples match the target
    if synthetic_data.shape[0] != num_samples_total:
        print(f"Warning: Generated {synthetic_data.shape[0]} samples, but target was {num_samples_total}.")

    return synthetic_data, synthetic_labels

# ==============================================
# == Synthetic Dataloader Helper Function     ==
# ==============================================
def get_synthetic_dataloader(cfg, synthetic_data, synthetic_labels):
    """Creates a DataLoader from synthetic data tensors."""
    if len(synthetic_data) == 0:
        print("Warning: Attempting to create DataLoader with 0 synthetic samples.")
        # Return an empty dataloader or handle appropriately
        return None
    synthetic_dataset = TensorDataset(synthetic_data, synthetic_labels)
    synthetic_dataloader = DataLoader(
        synthetic_dataset,
        batch_size=cfg.get("finetune_batch_size", cfg.batch_size), # Use finetune batch size or default
        shuffle=True,
        num_workers=getattr(cfg, 'num_workers', 0), # Often 0 for TensorDataset
        pin_memory=False # Usually False for TensorDataset
    )
    return synthetic_dataloader


# ==============================================
# ==      Federated Training Function         ==
# ==============================================
def train_federated(cfg: Config,
                    # Pass client train loaders dict and combined test loaders
                    client_train_loaders: Dict[int, DataLoader],
                    combined_test_loader: DataLoader,
                    mnist_test_loader: DataLoader,
                    fashion_test_loader: DataLoader,
                    model: nn.Module):

    # Create a dictionary to hold all test loaders for analysis functions
    dataloaders_for_analysis = {
        "mnist_test": mnist_test_loader,
        "fashion_test": fashion_test_loader,
        "combined_test": combined_test_loader
        # Add client train loaders if analysis needs them (might be large)
        # **{"client_" + str(cid): loader for cid, loader in client_train_loaders.items()}
    }
    num_gen_samples = cfg.num_gen_samples # Small number for logging
    # --- Client Initialization ---
    print(f"Initializing {cfg.num_clients} clients...")
    clients: List[Client] = []
    client_subset_assignment = {}
    for client_idx in range(cfg.num_clients):
        if client_idx not in client_train_loaders:
            print(f"Warning: No data loader found for client {client_idx}, skipping initialization.")
            continue
        # Pass the specific client's dataloader
        # Determine mu_target based on client subset assignment (needs access to that info)
        # For now, use a default or potentially lookup based on client_idx
        # This requires client_subset_assignment from split_data_federated
        # Let's assume a simple alternating pattern for mu_target for demo
        # TODO: Get actual subset assignment and map to mu_target correctly
        subset_id = client_idx // (cfg.num_clients // cfg.num_subsets) # Approximate subset id
        mu_target = 0#cfg.mnist_vae_mu_target if subset_id % 2 == 0 else cfg.fashion_vae_mu_target # Example logic

        client_info = {"client_idx": client_idx, "subset_id": subset_id} # Add more info if needed
        client_subset_assignment[client_idx] = subset_id # Store assignment for later use
        clients.append(
            get_client(
                cfg,
                deepcopy(model),
                client_train_loaders[client_idx], # The specific loader for this client
                mu_target, # Assign appropriate mu_target
                **client_info
            )
        )
    if len(clients) != cfg.num_clients:
         print(f"Warning: Initialized {len(clients)} clients, but cfg.num_clients is {cfg.num_clients}.")
         # Adjust num_clients if some were skipped? Or handle error.
         # For now, proceed with the initialized clients.

    if not clients:
        raise RuntimeError("No clients were initialized. Check data splitting and client configuration.")

    # Get Server
    print("Initializing server...")
    server = get_server(cfg, deepcopy(model)) # Server starts with the initial model

    num_rounds = cfg.num_rounds
    local_epochs = cfg.local_epochs
    clients_per_round = max(1, int(cfg.participation_rate * len(clients))) # Number of clients per round
    last_client_localmodels = [] # Store last local models for fine-tuning
    for client in clients:
        last_client_localmodels.append(deepcopy(client.model))

    # --- Federated Training Loop ---
    print(f"Starting federated training for {num_rounds} rounds with {len(clients)} total clients, {clients_per_round} per round...")
    for round_num in tqdm(range(num_rounds), desc="Federated Rounds"):
        wandb_round_results = {} # Reset results for this round's log
        figures_to_close = []

        # 1. Select Participating Clients for this round
        participating_client_indices = np.random.choice(
            range(len(clients)), clients_per_round, replace=False
        )
        participating_clients = [clients[i] for i in participating_client_indices]
        print(f"\nRound {round_num+1}/{num_rounds}, Participating clients: {participating_client_indices}")

        # 2. Distribute Global Model to Participating Clients
        global_state_dict = server.global_model.state_dict()
        for client in participating_clients:
            client.update_model(global_state_dict)

        # 3. Train Participating Clients Locally
        client_weights_updates = [] # Store results as (weight, update_dict) or similar
        round_local_losses = defaultdict(list) # Accumulate losses from participants

        for client in tqdm(participating_clients, desc=f"  Round {round_num+1} Local Train", leave=False):
            # Train returns state_dict and loss dict
            update_dict, local_loss_dict = client.train(local_epochs, round_num)
            client_weights_updates.append([update_dict]) # Append the returned update dict
            for key, value in local_loss_dict.items():
                round_local_losses[key].append(value) # Store loss values

            last_client_localmodels[client.client_idx] = deepcopy(client.model) # Store last local model

        # Log average local losses for the round
        for key, values in round_local_losses.items():
            wandb_round_results[f"train_local_avg_{key}"] = np.mean(values)

        # 4. Optional: Analyze Local Models (Before Aggregation)
        # Analyze only a subset of participating clients to save time/wandb space?
        if cfg.get("analyze_local_models_before_update", False):
             num_clients_to_analyze = len(participating_clients)
             print(f"  Analyzing {num_clients_to_analyze} local models...")
             for i in range(num_clients_to_analyze):
                 client_to_analyze = participating_clients[i]
                 client_idx = client_to_analyze.client_idx # Get the original client index
                 prefix = f"client{client_idx}_"
                 # Pass necessary client info if analyze_model needs it
                 client_info = {"client_idx": client_idx, "subset_id": client_to_analyze.subset_id}
                 analysis = analyze_model(cfg, client_to_analyze.model, f"Client {client_idx} Round {round_num+1} Pre-Agg",
                                          data_loaders=dataloaders_for_analysis, # Pass test loaders
                                          prefix=prefix,
                                          **client_info, # Pass client specific info if needed by analysis
                                          compare_model=server.global_model)
                 wandb_round_results, figures_to_close = log_analysis(wandb_round_results, analysis, figures_to_close)

        

        # 5. Aggregate Client Updates on Server
        print(f"  Aggregating updates from {len(client_weights_updates)} clients...")
        # Modify server.aggregate if it expects state dicts instead of weights/deltas
        # Assuming server.aggregate can handle the list of update dicts
        server.aggregate(client_weights_updates) # Pass the collected updates

        # 6. Analyze Aggregated Global Model
        print(f"  Analyzing aggregated global model...")
        server_analysis = analyze_model(cfg, server.global_model, f"Server Round {round_num+1}",
                                        data_loaders=dataloaders_for_analysis, # Use test loaders for server eval
                                        prefix="server_")
        wandb_round_results, figures_to_close = log_analysis(wandb_round_results, server_analysis, figures_to_close)

        # 7. Optional: Generate & Log Synthetic Data (Less Frequently?)
        # Generate synthetic data from a subset of clients or global model for monitoring
        if (round_num + 1) % cfg.log_synthetic_freq == 0:
            print("  Generating and logging synthetic data sample...")
            # Generate small sample using representative clients or global model
            # Need access to the specific client models involved *in this round*
            # Get models from participating clients
            last_local_models = [deepcopy(c.model) for c in participating_clients]

            # Adapt generate_synthetic_data to handle potentially fewer models or just use global
            
            #num_gen_clients = min(2, len(last_local_models)) # Use first 2 clients for example

            #if num_gen_clients >= 2:
            local_synthetic_data, local_synthetic_labels = generate_synthetic_data(
                cfg, client_models=last_local_models, 
                global_model=None, num_samples_total=num_gen_samples , device=cfg.device, gen_from_global=False, client_subset_assignment=client_subset_assignment, client_idxs=participating_client_indices
            )
            # elif num_gen_clients == 1:
            #      gen_model_0 = last_local_models[0]
            #      # Generate only one type or duplicate?
            #      local_synthetic_data, local_synthetic_labels = generate_synthetic_data(
            #          cfg, mnist_model=gen_model_0, fashion_model=gen_model_0, # Use same model twice
            #          global_model=None, num_samples_per_client=num_gen_samples // 2, device=cfg.device)
            # else: # No participating clients? Should not happen
            #      local_synthetic_data, local_synthetic_labels = torch.empty(0), torch.empty(0)


            if local_synthetic_data.numel() > 0:
                local_synthetic_dataloader = get_synthetic_dataloader(cfg, local_synthetic_data, local_synthetic_labels)
                if local_synthetic_dataloader:
                     # visualize_and_log_synthetic_data needs adjustment to return figure or handle wandb internally
                     # Assuming log_synthetic_batch_wandb returns a figure
                     local_synthetic_fig = log_synthetic_batch_wandb(local_synthetic_dataloader, caption=f"Local Synthetic_R{round_num+1}", num_samples=64)
                     wandb_round_results[f'local_synthetic'] = wandb.Image(local_synthetic_fig)
                     figures_to_close.append(local_synthetic_fig)

            # Generate from global model
            global_synthetic_data, global_synthetic_labels = generate_synthetic_data(
                 cfg, client_models=last_local_models, # No specific clients
                 global_model=server.global_model, num_samples_total=num_gen_samples, device=cfg.device, gen_from_global=True, client_subset_assignment=client_subset_assignment, client_idxs=participating_client_indices
            )

            if global_synthetic_data.numel() > 0:
                global_synthetic_dataloader = get_synthetic_dataloader(cfg, global_synthetic_data, global_synthetic_labels)
                if global_synthetic_dataloader:
                    global_synthetic_fig = log_synthetic_batch_wandb(global_synthetic_dataloader, caption=f"Global Synthetic_R{round_num+1}", num_samples=64)
                    wandb_round_results[f'global_synthetic'] = wandb.Image(global_synthetic_fig)
                    figures_to_close.append(global_synthetic_fig)


        # 8. Log Round Results to W&B
        wandb.log(wandb_round_results, step=round_num + 1)

        # 9. Clean up figures and memory
        for fig in figures_to_close:
            plt.close(fig)
        del client_weights_updates # Free memory from updates
        gc.collect()
        torch.cuda.empty_cache()


    print("Federated training finished.")

    # --- Optional: Save Final Global Model ---
    if cfg.save_dir is not None:
        save_path = f"{cfg.save_dir}/{cfg.name}.pth"
        torch.save(server.global_model.state_dict(), save_path)
        print(f"Global model saved to {save_path}")

    # --- Optional: Fine-tuning Phase ---
    if cfg.get("finetune_last_model"):
        print("\nStarting fine-tuning phase on synthetic data...")
        cfg_finalmodel = copy.deepcopy(cfg)
        cfg_finalmodel.model_name = getattr(cfg, "finetune_model_name", "vae")
        cfg_finalmodel.use_classifier = False
        cfg_finalmodel.client_classifier = False
        if cfg.get("finetune_kl_weight", -1) >= 0: # Allow overriding KL weight
             cfg_finalmodel.kl_weight = cfg.finetune_kl_weight

        # 1. Get Last *Participating* Client Models (More representative)
        # Use the models collected just before aggregation in the *last* round
        # Need to store these models explicitly if needed after the loop
        # For simplicity, re-using the logic, assuming 'clients' list holds final local states (potentially stale if participation < 1.0)
        # TODO: Improve this by storing models of last *participating* clients
        #print("WARNING: Fine-tuning using potentially stale client models if participation < 1.0")
        # Use the last local models from the loop

        # 2. Generate Synthetic Data
        synthetic_data, synthetic_labels = generate_synthetic_data(
            cfg, # Use final model config
            client_models=last_client_localmodels,
            global_model=server.global_model,
            num_samples_total=cfg.get("finetune_num_samples", 100000), # Total samples for fine-tuning
            device=cfg.device, 
            gen_from_global=cfg.finetune_gen_global_model,
            client_subset_assignment=client_subset_assignment, # Pass the subset assignment
        )

        # 3. Create Synthetic Dataloader
        synthetic_dataloader = get_synthetic_dataloader(cfg, synthetic_data, synthetic_labels)
        if not synthetic_dataloader:
             print("Error: Could not create synthetic dataloader for fine-tuning. Skipping.")
             return server.global_model # Return the original federated model

        wandb_round_results = {}

        synthetic_image = log_synthetic_batch_wandb(synthetic_dataloader, caption = "synthetic_data", num_samples=64)
        wandb_round_results['synthetic_data'] = wandb.Image(synthetic_image)
        wandb.log(wandb_round_results, step=num_rounds)
        plt.close(synthetic_image)




        # 4. Visualize Synthetic Data (Once)
        #visualize_and_log_synthetic_data(synthetic_dataloader, num_samples=64, caption="Synthetic Data for Finetuning")

        # 5. Initialize Unconditional Model
        unconditional_model = get_model(cfg_finalmodel)
        if cfg.get("finetune_init_from_global", False):
             print("  Initializing fine-tuning model with final global server weights.")
             # Use strict=False if architecture differs (e.g., removed classifier)
             missing_keys, unexpected_keys = unconditional_model.load_state_dict(server.global_model.state_dict(), strict=False)
             if missing_keys: print(f"  Warning: Missing keys during fine-tune init: {missing_keys}")
             if unexpected_keys: print(f"  Warning: Unexpected keys during fine-tune init: {unexpected_keys}")
        unconditional_model.to(cfg.device)

        # 6. Fine-tune using a simple client setup
        print("  Setting up fine-tuning client...")
        finetuning_client_cfg = copy.deepcopy(cfg) # Start from original cfg
        finetuning_client_cfg.client_type = "base"
        finetuning_client_cfg.local_epochs = 1
        finetuning_client_cfg.use_classifier = False
        finetuning_client_cfg.client_classifier = False
        finetuning_client_cfg.lr = cfg.get("finetune_lr", cfg.lr)
        finetuning_client_cfg.mu_target = 0 # Target N(0,I)
        finetuning_client_cfg.kl_weight = cfg_finalmodel.kl_weight # Use potentially overridden KL weight

        finetuning_client = get_client(
            finetuning_client_cfg,
            unconditional_model, # Pass the model instance to be fine-tuned
            synthetic_dataloader,
            0 # mu_target=0
        )

        finetune_epochs = cfg.get("finetune_epochs", 100)
        print(f"  Starting fine-tuning for {finetune_epochs} epochs...")
        for epoch in tqdm(range(finetune_epochs), desc="Finetuning Epochs"):
            ft_wandb_results = {}
            ft_figures_to_close = []

            # Train for one epoch - model is updated in-place within the client
            _, ft_loss_dict = finetuning_client.train(1)
            for key, value in ft_loss_dict.items():
                ft_wandb_results[f"finetune_{key}"] = value # Log fine-tuning loss

            # Analyze fine-tuned model (optional, maybe less frequent)
            if (epoch + 1) % cfg.analyze_finetuned_freq == 0:
                finetuned_analysis = analyze_model(
                    finetuning_client_cfg, # Use fine-tuning config
                    finetuning_client.model,
                    f"Finetuned Epoch {epoch+1}",
                    data_loaders=dataloaders_for_analysis, # Analyze on original test data
                    prefix="finetuned_"
                )
                ft_wandb_results, ft_figures_to_close = log_analysis(ft_wandb_results, finetuned_analysis, ft_figures_to_close)

            # Log synthetic data from fine-tuned model (optional, maybe less frequent)
            pseudo_client_models = [deepcopy(finetuning_client.model) for element in last_client_localmodels]
            if (epoch + 1) % cfg.log_finetuned_synthetic_freq == 0:
                 finetuned_synthetic_data, finetuned_synthetic_labels = generate_synthetic_data(
                     finetuning_client_cfg, client_models=pseudo_client_models, 
                     global_model=finetuning_client.model, num_samples_total=num_gen_samples,
                     device=cfg.device, gen_from_global=True,
                     client_subset_assignment=client_subset_assignment, # Pass the subset assignment
                 )
                 if finetuned_synthetic_data.numel() > 0:
                      ft_synth_loader = get_synthetic_dataloader(cfg, finetuned_synthetic_data, finetuned_synthetic_labels)
                      if ft_synth_loader:
                           ft_synth_fig = log_synthetic_batch_wandb(ft_synth_loader, caption=f"Finetuned Synth E{epoch+1}", num_samples=64)
                           ft_wandb_results[f'finetuned_synthetic'] = wandb.Image(ft_synth_fig)
                           ft_figures_to_close.append(ft_synth_fig)


            # Log fine-tuning results (offset step)
            wandb.log(ft_wandb_results, step=num_rounds + epoch + 1)

            # Clean up figures
            for fig in ft_figures_to_close: plt.close(fig)
            gc.collect()
            torch.cuda.empty_cache()

        print("Finished fine-tuning.")
        final_finetuned_model = finetuning_client.model # Get the final trained model

        if cfg.get("save_finetuned_model", True) and cfg.save_dir is not None:
             finetuned_save_path = f"{cfg.save_dir}/{cfg.name}_finetuned.pth"
             torch.save(final_finetuned_model.state_dict(), finetuned_save_path)
             print(f"Fine-tuned model saved to {finetuned_save_path}")

        return final_finetuned_model # Return the fine-tuned model

    # Return the final federated model if not fine-tuning
    return server.global_model


# ==============================================
# ==      Main Execution Block                ==
# ==============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="federated_run", help="Experiment name for config and wandb")
    # Add arguments to override config values if needed
    # e.g., parser.add_argument("--num_clients", type=int)
    #       parser.add_argument("--split_mode", type=str) ...
    args = parser.parse_args()

    # Load configuration
    cfg = get_config(exp_name=args.name)
    # --- Override config with command-line args if provided ---
    # Example: if args.num_clients is not None: cfg.num_clients = args.num_clients
    # ... implement overrides for other relevant args ...

    # Basic Config Validation
    if not hasattr(cfg, 'num_clients') or cfg.num_clients <= 0:
        raise ValueError("Configuration must define 'num_clients' > 0")
    if not hasattr(cfg, 'split_mode'):
        raise ValueError("Configuration must define 'split_mode' (e.g., 'iid', 'noniid-class', 'noniid-dirichlet')")
    # Add checks for mode-specific required parameters (num_subsets, num_classes_per_client, alpha)

    # Print experiment information (improved)
    print("--- Experiment Configuration ---")
    config_dict = asdict(cfg) if hasattr(cfg, 'asdict') else vars(cfg)
    for key, value in config_dict.items():
        print(f"  {key}: {value}")
    print("-------------------------------")




    # Get datasets and dataloaders using the new federated function
    # Assuming cfg contains necessary parameters like data_dir, batch_size etc.
    # The function now returns client_train_loaders dict and test loaders
    client_train_loaders, combined_test_loader, mnist_test_loader, fashion_test_loader, filename_parts = get_dataloaders_federated(cfg)


    # Initialize W&B
    wandb_mode = "online" if getattr(cfg, 'wandb', True) else "disabled" # Default to enabled if not set
    try:
        wandb.init(
            entity="FedRL-SNU", # Your entity
            project=getattr(cfg, 'project', 'VAE-Federated'), # Default project name
            name=cfg.name + "_" + "_".join(filename_parts), # Experiment name
            config=config_dict, # Log the full config
            mode=wandb_mode
        )
        print(f"Wandb initialized in '{wandb_mode}' mode.")
    except Exception as e:
        print(f"Error initializing W&B: {e}. Check entity/project name.")
        # Decide how to proceed, e.g., exit or continue without wandb
        wandb_mode = "disabled" # Fallback to disabled


    if not client_train_loaders:
         print("Error: No client training dataloaders were created. Exiting.")
         if wandb_mode != "disabled": wandb.finish(exit_code=1)
         exit()


    # Get Model
    print("Initializing model...")
    model = get_model(cfg)

    # Run Federated training
    final_model = train_federated(
        cfg,
        client_train_loaders,
        combined_test_loader,
        mnist_test_loader,
        fashion_test_loader,
        model
    )

    if wandb_mode != "disabled": wandb.finish()
    print("Run finished.")