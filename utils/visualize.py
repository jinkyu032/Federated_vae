import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from torch.utils.data import DataLoader
from typing import Dict
import torch.nn as nn
from collections import defaultdict
import gc # Garbage collector for explicit memory management
from sklearn.feature_selection import mutual_info_classif

__all__ = ['plot_latent_space', 'plot_latent_per_client', 'plot_latent_per_client', 'plot_recontruction_from_noise', 'analyze_latent_space']

# Get latent representations
def get_latent_representations(model, data_loader, device=None):
    model.eval()
    mus, labels = [], []
    with torch.no_grad():
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            _, mu, _, _ = model(data, targets)
            mus.append(mu.cpu().numpy())
            labels.append(targets.cpu().numpy())
    return np.concatenate(mus, axis=0), np.concatenate(labels, axis=0)

# Plot latent space
def plot_latent_space(model, mnist_loader, fashion_loader, title, device=None):
    mnist_mus, mnist_labels = get_latent_representations(model, mnist_loader, device)
    fashion_mus, fashion_labels = get_latent_representations(model, fashion_loader, device)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(mnist_mus[:, 0], mnist_mus[:, 1], c=mnist_labels, cmap='tab10', marker='o', s=10, alpha=0.5, label='MNIST')
    ax.scatter(fashion_mus[:, 0], fashion_mus[:, 1], c=fashion_labels, cmap='tab10', marker='x', s=10, alpha=0.5, label='Fashion-MNIST')
    
    ax.set_title(title)
    ax.set_xlabel('Latent Dim 1')
    ax.set_ylabel('Latent Dim 2')
    ax.legend()
    cbar = fig.colorbar(ax.collections[0], ax=ax, ticks=np.arange(10))
    cbar.set_label('Label (0-9)')
    return fig

def plot_latent_per_client(mnist_model, fashion_model, data_loaders: Dict[str, DataLoader], title=None, device=None):
    mnist_test_loader = data_loaders["mnist_test"]
    fashion_test_loader = data_loaders["fashion_test"]

    mnist_mus, mnist_labels = get_latent_representations(mnist_model, mnist_test_loader, device)
    fashion_mus, fashion_labels = get_latent_representations(fashion_model, fashion_test_loader, device)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(mnist_mus[:, 0], mnist_mus[:, 1], c=mnist_labels, cmap='tab10', marker='o', s=10, alpha=0.5, label='MNIST')
    ax.scatter(fashion_mus[:, 0], fashion_mus[:, 1], c=fashion_labels, cmap='tab10', marker='x', s=10, alpha=0.5, label='Fashion-MNIST')
    
    ax.set_title(title)
    ax.set_xlabel('Latent Dim 1')
    ax.set_ylabel('Latent Dim 2')
    ax.legend()
    cbar = fig.colorbar(ax.collections[0], ax=ax, ticks=np.arange(10))
    cbar.set_label('Label (0-9)')
    return fig


#From bvezilic/Variational-autoencoder
class PlotCallback:
    """Callback class that retrieves several samples and displays model reconstructions"""
    def __init__(self, cfg, num_samples=10, save_dir=None, device=None):
        self.num_samples = num_samples
        self.device = device
        self.cfg = cfg
        self.num_total_classes = cfg.num_total_classes
        # self.save_dir = save_dir
        # self.counter = 0

        # if self.save_dir and not os.path.exists(self.save_dir):
        #     os.makedirs(self.save_dir)

    def __call__(self, model, dataloader):
        model.eval()  # Set model to eval mode due to Dropout, BN, etc.
        with torch.no_grad():
            inputs, targets = self._batch_random_samples(dataloader)
            outputs, mu, log_var, z = model(inputs, targets)  # Forward pass

            # Prepare data for plotting
            input_images = self._reshape_to_image(inputs, numpy=True)
            recon_images = self._reshape_to_image(outputs, numpy=True)
            z_ = self._to_numpy(z)

            fig = self._plot_samples(input_images, recon_images, z_)

        model.train()  # Return to train mode
        return fig

    def _batch_random_samples(self, dataloader):
        """Helper function that retrieves `num_samples` from dataset and prepare them in batch for model """
        dataset = dataloader.dataset

        # Randomly sample one data sample per each class
        samples = []
        targets = []
        for i in range(self.num_total_classes):
            idxs = np.where(dataset.targets == i)[0]
            if len(idxs) == 0:
                continue
            idx = np.random.choice(idxs, size=1)
            #breakpoint()
            samples.append(dataset[idx[0]][0])
            targets.append(i)

        # Create batch
        batch = torch.stack(samples)
        batch = batch.view(batch.size(0), -1)  # Flatten
        batch = batch.to(self.device)

        targets = torch.tensor(targets, dtype=torch.long).to(self.device)

        return batch, targets

    def _reshape_to_image(self, tensor, numpy=True):
        """Helper function that converts image-vector into image-matrix."""
        images = tensor.reshape(-1, 28, 28)
        if numpy:
            images = self._to_numpy(images)

        return images

    def _to_numpy(self, tensor):
        """Helper function that converts tensor to numpy"""
        return tensor.cpu().numpy()

    def _plot_samples(self, input_images, recon_images, z):
        """Creates plot figure and saves it on disk if save_dir is passed."""
        fig, ax_lst = plt.subplots(self.num_samples, 3)
        fig.suptitle("Input → Latent Z → Reconstructed")

        for i in range(self.num_samples):
            # Images
            ax_lst[i][0].imshow(input_images[i], cmap="gray")
            ax_lst[i][0].set_axis_off()

            # Variable z
            ax_lst[i][1].bar(np.arange(len(z[i])), z[i])

            # Reconstructed images
            ax_lst[i][2].imshow(recon_images[i], cmap="gray")
            ax_lst[i][2].set_axis_off()

        #fig.tight_layout()
        return fig
    
def plot_recontruction_from_noise(cfg, model, num_samples=10, device=None, mu=0, std=1):
    model.eval()
    with torch.no_grad():
        z = torch.normal(mean=mu, std=std, size=(num_samples, cfg.latent_dim)).to(device)
        # unconditional reconstruction
        ## TODO: conditional reconstruction
        outputs = model.decoder_forward(z)
        images = outputs.reshape(-1, 28, 28)
        z = z.cpu().numpy()
        images = images.cpu().numpy()

        """Creates plot figure and saves it on disk if save_dir is passed."""
        fig, ax_lst = plt.subplots(num_samples, 3)
        fig.suptitle("Latent Z → Reconstructed")

        for i in range(num_samples):
            # Variable z            
            ax_lst[i][1].bar(np.arange(len(z[i])), z[i])

            # Reconstructed images
            ax_lst[i][2].imshow(images[i], cmap="gray")
            ax_lst[i][2].set_axis_off()

        return fig


# visualize the latent space
def visualize_manifold(model, num_samples=20, device=None, offset = (0, 0), do = True):
    if not do:
        return None
    x = norm.ppf(np.linspace(0.011, 0.99, num_samples)) + offset[0]
    y = norm.ppf(np.linspace(0.011, 0.99, num_samples)) + offset[1]
    inputs = [(i, j) for i in x for j in y]
    inputs_t = torch.tensor(inputs, dtype=torch.float32).to(device)  
    model.eval()  
    with torch.no_grad():
        #outputs = model.decoder(inputs_t)
        outputs = model.decoder_forward(inputs_t)
    outputs = outputs.cpu().numpy()
    outputs = outputs.reshape(num_samples, num_samples, 28, 28)
    fig, axes = plt.subplots(num_samples, num_samples, figsize=(12, 12), subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(0, 0)
    for i in range(outputs.shape[0]):
        for j in range(outputs.shape[1]):
            axes[i][j].imshow(outputs[i][j], cmap='gray')
            
            if j == 0:
                axes[i][j].set_ylabel("{:.2f}".format(y[i]))
            if i == 0:
                axes[i][j].set_xlabel("{:.2f}".format(x[j]))
                axes[i][j].xaxis.set_label_position('top') 
    return fig





# --- VAE Loss Function (Placeholder - Needs implementation) ---
# def vae_loss(recon_x, x, mu, log_var, kl_weight=1.0):
#     # ... (Your implementation)
#     pass

# --- Latent Space Analysis Function (cuML version for Class-Specific MI) ---

def analyze_latent_space(cfg, model, data_loader, device, **kwargs):
    """
    Analyzes the latent space using scikit-learn for CLASS-SPECIFIC MI.

    Calculates and visualizes:
    1. Per-class, per-channel Mean and Variance (Streaming on CPU).
    2. Per-class, per-channel Mean Absolute Activation (Streaming on CPU).
    3. Per-channel, PER-CLASS Mutual Information (CPU using scikit-learn).
       MI(channel_k; is_class_c) for each channel k and class c.

    Args:
        cfg: Configuration object. Must contain attributes like num_classes,
             latent_dim, use_classifier, kl_weight, etc.
        model: The trained VAE model.
        data_loader: DataLoader for the dataset to analyze.
        device: The device to run model inference on (e.g., 'cuda' or 'cpu').
        **kwargs: Placeholder for potential future args, matches signature if
                  this replaced the original compute_loss.

    Returns:
        dict: Calculated statistics:
              'class_channel_stats': {class: {channel: {'mean', 'variance', 'mean_abs'}}}
              'mi_scores_per_class': (num_classes, latent_dim) numpy array of MI scores.
              'losses': dict of average losses ('total_loss', 'recon_loss', 'kl_loss', 'accuracy').
        dict: Matplotlib figures:
              'mean_heatmap', 'variance_heatmap',
              'mean_abs_heatmap', 'mi_per_class_heatmap'
    """
    # --- Extract parameters from cfg or set defaults ---
    num_classes = getattr(cfg, 'num_classes', 20) # Default e.g. 20
    latent_dim = getattr(cfg, 'latent_dim', 22) # Default e.g. 20
    use_mu_for_analysis = getattr(cfg, 'use_mu_for_analysis', True)
    mi_n_neighbors = getattr(cfg, 'mi_n_neighbors', 5)
    mi_subsample_ratio = getattr(cfg, 'mi_subsample_ratio', None)
    calculate_losses = getattr(cfg, 'calculate_losses', True) # Assume losses needed
    kl_weight = getattr(cfg, 'kl_weight', 1.0) # Default KL weight
    mu_target_vae = getattr(cfg, 'mu_target', 0) # mu_target for VAE loss

    # Removed CUDA device check specific to cuML

    model.eval()
    model.to(device)

    # --- Data Structures for Streaming Statistics (on CPU) ---
    stats_accumulator = defaultdict(lambda: defaultdict(lambda: {'sum': 0.0, 'sum_sq': 0.0, 'sum_abs': 0.0, 'count': 0, 'latent_var': 0.0}))

    # --- Data Storage for MI (Lists of NumPy Arrays) ---
    all_latents_cpu_list = []
    all_targets_cpu_list = []

    # --- Loss Accumulators ---
    total_loss_sum = 0
    recon_loss_sum = 0
    kl_loss_sum = 0
    total_samples = 0
    correct = 0 # For accuracy if classifier is used

    #print("Starting latent space analysis (Class-Specific MI using scikit-learn)...")
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            # Keep target on CPU for numpy indexing and sklearn
            target_cpu = target.numpy()
            target = target.to(device) # Keep on GPU if model needs it

            batch_size = data.size(0)
            total_samples += batch_size

            # --- VAE Forward Pass ---
            try:
                # Adapt based on your VAE model's forward signature
                if getattr(cfg, 'use_classifier', False):
                    recon_batch, mu, log_var, z, class_output = model(data, return_classfier_output=True)
                    # _, predicted = torch.max(class_output.data, 1)
                    # correct += (predicted == target_gpu).sum().item()
                else:
                    # If model takes target, pass target_gpu
                    # If model only takes data, call model(data)
                    recon_batch, mu, log_var, z = model(data, target) # Assuming model(data) if no classifier
            except Exception as e:
                 print(f"Error during model forward pass: {e}")
                 # Decide how to handle: return None, raise error, etc.
                 return None, None # Example: return None if forward pass fails

            # --- Select Vector for Analysis ---
            vectors_to_analyze_gpu = mu if use_mu_for_analysis else z
            vectors_cpu = vectors_to_analyze_gpu.detach().cpu().numpy()
            var_cpu = torch.exp(log_var).detach().cpu().numpy() # Convert log_var to variance

            # --- Loss Calculation ---
            # if calculate_losses:
            #     try:
            #         # Use the specific vae_loss function defined above
            #         # Pass mu_target_vae to the loss function
            #         recon_loss, kl_loss = vae_loss(recon_batch, data, mu, log_var, mu_target=mu_target_vae, reduction='sum') # Get sum for batch
            #         # Accumulate batch sums
            #         total_loss_sum += (recon_loss.item() + kl_weight * kl_loss.item())
            #         recon_loss_sum += recon_loss.item()
            #         kl_loss_sum += kl_loss.item()
            #     except Exception as e:
            #         print(f"Warning: Error during loss calculation: {e}. Skipping loss calculation.")
            #         calculate_losses = False

            # --- Accumulate Streaming Statistics (on CPU) ---
            for i in range(batch_size):
                class_label = target_cpu[i]
                current_vector = vectors_cpu[i]
                for k in range(latent_dim):
                    val = current_vector[k]
                    stats_accumulator[class_label][k]['sum'] += val
                    stats_accumulator[class_label][k]['sum_sq'] += val**2
                    stats_accumulator[class_label][k]['sum_abs'] += abs(val)
                    stats_accumulator[class_label][k]['count'] += 1
                    stats_accumulator[class_label][k]['latent_var'] += var_cpu[i][k] # Accumulate log_var for each class and channel

            # --- Store CPU Data for MI ---
            all_latents_cpu_list.append(vectors_cpu)
            all_targets_cpu_list.append(target_cpu)

            # if batch_idx % 50 == 0 and batch_idx > 0:
            #      print(f"  Processed batch {batch_idx}/{len(data_loader)}")
            #      # Optional: Clean up GPU tensors from loop if memory is tight on GPU
            #      # del recon_batch, mu, log_var, z, vectors_to_analyze_gpu, class_output # if use_classifier
            #      # gc.collect(); torch.cuda.empty_cache()


    print("Finished data iteration. Calculating final statistics...")

    # --- Finalize Streaming Statistics Calculation (on CPU) ---
    results = {'class_channel_stats': defaultdict(lambda: defaultdict(dict))}
    for c in range(num_classes):
        for k in range(latent_dim):
            count = stats_accumulator[c][k]['count']
            if count > 0:
                mean = stats_accumulator[c][k]['sum'] / count
                mean_sq = stats_accumulator[c][k]['sum_sq'] / count
                variance = max(0.0, mean_sq - (mean**2))
                mean_abs = stats_accumulator[c][k]['sum_abs'] / count
                latent_var = stats_accumulator[c][k]['latent_var'] / count
            else:
                mean, variance, mean_abs, latent_var = np.nan, np.nan, np.nan, np,nan
            results['class_channel_stats'][c][k]['mean'] = mean
            results['class_channel_stats'][c][k]['variance'] = variance
            results['class_channel_stats'][c][k]['mean_abs'] = mean_abs
            results['class_channel_stats'][c][k]['latent_var'] = latent_var


    # results['total_loss'] = total_loss_sum / total_samples
    # results['recon_loss'] = recon_loss_sum / total_samples
    # results['kl_loss'] = kl_loss_sum / total_samples
    # if getattr(cfg, 'use_classifier', False):
    #     results['accuracy'] = 100.0 * correct / total_samples

    # --- Calculate Class-Specific Mutual Information using scikit-learn ---
   # print("Calculating Class-Specific Mutual Information using scikit-learn...")
    # mi_scores_per_class_np = np.full((num_classes, latent_dim), np.nan)
    # all_latents_np = None
    # all_targets_np = None

    # try:
    #     # 1. Concatenate NumPy arrays
    #     print("  Concatenating NumPy arrays...")
    #     all_latents_np = np.concatenate(all_latents_cpu_list, axis=0)
    #     all_targets_np = np.concatenate(all_targets_cpu_list, axis=0)
    #     del all_latents_cpu_list, all_targets_cpu_list # Free list memory
    #     gc.collect()
    #     print(f"  Concatenated latent shape: {all_latents_np.shape}")

    #     # --- Optional Subsampling for MI (on NumPy array) ---
    #     current_latents_np = all_latents_np
    #     current_targets_np = all_targets_np
    #     if mi_subsample_ratio is not None and 0 < mi_subsample_ratio < 1:
    #         num_total_samples = current_latents_np.shape[0]
    #         num_subsamples = int(num_total_samples * mi_subsample_ratio)
    #         print(f"  Subsampling {num_subsamples}/{num_total_samples} (NumPy) for MI calculation.")
    #         indices = np.random.choice(num_total_samples, num_subsamples, replace=False)
    #         current_latents_np = current_latents_np[indices]
    #         current_targets_np = current_targets_np[indices]

    #     if current_targets_np.shape[0] < mi_n_neighbors:
    #          print(f"Warning: Not enough samples ({current_targets_np.shape[0]}) for MI calculation with k={mi_n_neighbors}. Skipping MI.")
    #     else:
    #         # 3. Loop through each class and each channel
    #         for c in range(num_classes):
    #             print(f"  Calculating MI for class {c} vs Rest...")
    #             # Create binary target: 1 if class == c, 0 otherwise
    #             binary_targets_np = (current_targets_np == c).astype(np.int32)

    #             # Check if binary target has both 0s and 1s
    #             unique_binary_vals, counts = np.unique(binary_targets_np, return_counts=True)
    #             if len(unique_binary_vals) < 2:
    #                 print(f"    Skipping MI for class {c}: Only one value ({unique_binary_vals}) present in binary target.")
    #                 mi_scores_per_class_np[c, :] = 0.0
    #                 continue

    #             # Check if either class in binary target has fewer samples than n_neighbors
    #             min_samples_per_binary_class = min(counts)
    #             if min_samples_per_binary_class < mi_n_neighbors:
    #                 print(f"    Warning: Skipping MI for class {c}: Minimum samples in binary split ({min_samples_per_binary_class}) is less than n_neighbors ({mi_n_neighbors}).")
    #                 # Assign NaN or 0 based on preference
    #                 mi_scores_per_class_np[c, :] = np.nan # Or 0.0
    #                 continue


    #             # Loop through each channel for sklearn
    #             for k in range(latent_dim):
    #                 try:
    #                     # Extract current channel, reshape for sklearn
    #                     z_k = current_latents_np[:, k].reshape(-1, 1)

    #                     # Check variance for the channel
    #                     if np.var(z_k) < 1e-9:
    #                          #print(f"    Skipping MI for class {c}, channel {k} due to zero variance.")
    #                          mi_scores_per_class_np[c, k] = 0.0
    #                          continue

    #                     # Calculate MI for the specific channel k vs binary target c
    #                     mi = mutual_info_classif(
    #                         z_k,
    #                         binary_targets_np,
    #                         n_neighbors=mi_n_neighbors,
    #                         discrete_features=False, # Input channel is continuous
    #                         random_state=42
    #                     )
    #                     mi_scores_per_class_np[c, k] = mi[0] # Store the single score

    #                 except Exception as e_inner:
    #                     print(f"    Error calculating MI for class {c}, channel {k}: {e_inner}")
    #                     mi_scores_per_class_np[c, k] = np.nan # Assign NaN on error

    #             # Optional: Print progress per class if needed
    #             # if c % 1 == 0 or c == num_classes - 1:
    #             #     print(f"    Calculated MI for all channels for class {c}")


    #         print("  Class-Specific MI calculations complete.")

    # except MemoryError:
    #     print("MemoryError during MI calculation! Try using 'mi_subsample_ratio' or reduce data size. Skipping MI.")
    #     # mi_scores_per_class_np remains NaN array
    # except Exception as e_outer:
    #     print(f"An outer error occurred during MI preparation/calculation: {e_outer}")
    #     # mi_scores_per_class_np remains NaN array
    # finally:
    #     # Clean up large NumPy arrays
    #     del all_latents_np, all_targets_np, current_latents_np, current_targets_np # noqa F821
    #     gc.collect()

    # results['mi_scores_per_class'] = mi_scores_per_class_np

    figures = {}
    try:
        figures['mean_heatmap'] = plot_stat_heatmap(results['class_channel_stats'], 'mean', num_classes, latent_dim, "Mean Activation per Class/Channel")
        figures['variance_heatmap'] = plot_stat_heatmap(results['class_channel_stats'], 'variance', num_classes, latent_dim, "Variance per Class/Channel")
        figures['mean_abs_heatmap'] = plot_stat_heatmap(results['class_channel_stats'], 'mean_abs', num_classes, latent_dim, "Mean Abs Activation per Class/Channel")
        figures['latent_var_heatmap'] = plot_stat_heatmap(results['class_channel_stats'], 'latent_var', num_classes, latent_dim, "Latent Variance per Class/Channel")
        #figures['mi_per_class_heatmap'] = plot_mi_per_class_heatmap(results['mi_scores_per_class'], num_classes, latent_dim)
        print("Plots generated.")
    except Exception as e:
        print(f"An error occurred during plot generation: {e}")

    for key in figures:
        if isinstance(figures[key], plt.Figure):
            results[key] = figures[key]
            figures[key] = None # Clear figure from memory
            plt.close(figures[key]) # Close the figure to free memory

    # Print results
    # print("Results:")
    # print(results)
    # Return both results and figures
    return results


# --- Plotting Helper Functions (Remain the same, use NumPy arrays) ---

def plot_stat_heatmap(class_channel_stats, stat_key, num_classes, latent_dim, title):
    stat_matrix = np.full((num_classes, latent_dim), np.nan)
    for c in range(num_classes):
        for k in range(latent_dim):
            # Check if class and channel stats exist before accessing
            if c in class_channel_stats and k in class_channel_stats.get(c, {}):
                 stat_matrix[c, k] = class_channel_stats[c][k].get(stat_key, np.nan)

    fig, ax = plt.subplots(figsize=(max(8, latent_dim * 0.4), max(5, num_classes * 0.3)))
    cmap = plt.cm.viridis
    cmap.set_bad(color='grey') # Set color for NaN values

    im = ax.imshow(stat_matrix, aspect='auto', cmap=cmap, interpolation='nearest')
    ax.set_xlabel('Latent Dimension Index')
    ax.set_ylabel('Class Label')
    ax.set_xticks(np.arange(latent_dim))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(np.arange(latent_dim))
    ax.set_yticklabels(np.arange(num_classes))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    ax.set_title(title)
    plt.colorbar(im, label=stat_key.replace('_', ' ').title())
    plt.tight_layout()
    return fig

def plot_mi_per_class_heatmap(mi_scores_per_class, num_classes, latent_dim):
    """Generates a heatmap for class-specific MI scores."""
    fig, ax = plt.subplots(figsize=(max(8, latent_dim * 0.4), max(5, num_classes * 0.3)))

    # Handle potential NaNs visually for plotting
    # Use mask to handle NaNs with imshow's cmap
    masked_array = np.ma.masked_invalid(mi_scores_per_class) # Mask NaN values
    cmap = plt.cm.viridis
    cmap.set_bad(color='grey') # Color for masked (NaN) values

    # Plot using the masked array, vmin=0 to ensure non-negative range makes sense
    im = ax.imshow(masked_array, aspect='auto', cmap=cmap, interpolation='nearest', vmin=0)

    ax.set_xlabel('Latent Dimension Index')
    ax.set_ylabel('Class Label')
    ax.set_xticks(np.arange(latent_dim))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(np.arange(latent_dim))
    ax.set_yticklabels(np.arange(num_classes))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    ax.set_title('Mutual Information(Channel k; Is Class c?)')
    plt.colorbar(im, label='Mutual Information Score')
    plt.tight_layout()
    return fig