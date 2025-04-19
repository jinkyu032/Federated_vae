import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import gc # Garbage collector for explicit memory management
from sklearn.feature_selection import mutual_info_classif
import torch_fidelity
from torch_fidelity import calculate_metrics
from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Dict, Tuple, List, Optional
import math
from collections import defaultdict
from torchvision.utils import make_grid
import os
from torch.utils.data import ConcatDataset

__all__ = ['vae_loss', 'compute_loss', 'calculate_fid_is']

# VAE loss function
def vae_loss(recon_x, x, mu = None, log_var = None, mu_target=0, reduction='mean', reconloss_only=False):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = None
    if not reconloss_only:
        KLD = -0.5 * torch.sum(2*mu_target*mu + 1 + log_var - mu.pow(2) - log_var.exp() - mu_target*mu_target)

    if reduction == 'mean':
        BCE /= x.size(0)
        if not reconloss_only:
            KLD /= x.size(0)

    return BCE, KLD

def celoss(x,y, reduction='mean', temp=1):
    criterion = nn.CrossEntropyLoss(reduction=reduction)
    loss = criterion(x/temp, y)
    return loss

# Compute loss for a data loader
def compute_loss(cfg, model, data_loader, device, mu_target=0, reduction='sum', compare_model = None, *args, **kwargs):
    model_list = []
    model.eval()
    model.to(device)
    total_loss_sum = 0
    recon_loss_sum = 0
    kl_loss_sum = 0
    correct = 0
    codebook_loss_sum = 0
    commitment_loss_sum = 0
    client_class_celoss = 0
    model_list.append(model)
    mean_var = 0
    if compare_model is not None:
        compare_model.eval()
        compare_model.to(device)
        mean_l2_compare = 0
        var_diff_compare = 0
        model_list.append(compare_model)
        mean_var_compare = 0
    

    def forward_modellist(model_list, *args, **kwargs):
        result_list = []
        for model in model_list:
            result = model(*args, **kwargs)
            result_list.append(result)
        return result_list



    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            if cfg.vq:
                #recon_batch, codebook_loss, commitment_loss = model(data, target)
                result = model(data, target)
                recon_batch, codebook_loss, commitment_loss = result['recon_x'], result['codebook_loss'], result['commitment_loss']
                recon_loss, _ = vae_loss(recon_batch, data, reconloss_only=True, reduction=reduction)
                if reduction == 'mean':
                    codebook_loss = codebook_loss.mean()
                    commitment_loss = commitment_loss.mean()
                elif reduction == 'sum':
                    codebook_loss = codebook_loss.sum()
                    commitment_loss = commitment_loss.sum()
                total_loss_sum += (recon_loss.item() + codebook_loss.item() + cfg.commitment_weight * commitment_loss.item())
                recon_loss_sum += recon_loss.item()
                codebook_loss_sum += codebook_loss.item()
                commitment_loss_sum += commitment_loss.item()
            else:
                if cfg.client_classifier:
                    if kwargs.get("oracle", False):
                        results = forward_modellist(model_list, data, kwargs["client_idx"], return_classfier_output=True) #model(data, kwargs["client_idx"], return_classfier_output=True)
                    else:
                        results = forward_modellist(model_list, data, return_classfier_output=True)#model(data, return_classfier_output=True)

                    result = results[0]
                    #recon_batch, mu, log_var, z, class_output = result['recon_x'], result['mu'], result['log_var'], result['z'], result['client_class_output']
                    recon_batch, mu, log_var, z, prob_class_output, client_idxs = result['recon_x'], result['mu'], result['log_var'], result['z'], result['prob_class_output'],  result['client_idxs']

                    client_target = torch.ones(data.size(0),dtype=torch.long).to(cfg.device) * kwargs["client_idx"]
                    embedding_loss = celoss(prob_class_output, client_target, reduction=cfg.reduction)
                    client_class_celoss += embedding_loss.sum().item()
                    predicted = client_idxs
                    correct += (predicted == kwargs["client_idx"]).sum().item()
                    total_loss_sum += embedding_loss.sum().item()


                elif cfg.use_classifier:
                    #recon_batch, mu, log_var, z, class_output = model(data, return_classfier_output=True)
                    #result = model(data, return_classfier_output=True)
                    results = forward_modellist(model_list, data, return_classfier_output=True)
                    result = results[0]
                    recon_batch, mu, log_var, z, class_output = result['recon_x'], result['mu'], result['log_var'], result['z'], result['client_class_output']
                    _, predicted = torch.max(class_output.data, 1)
                    correct += (predicted == target).sum().item()
                else:
                    #recon_batch, mu, log_var, z = model(data, target)
                    #result = model(data, target)
                    results = forward_modellist(model_list, data, target)
                    result = results[0]
                    recon_batch, mu, log_var, z = result['recon_x'], result['mu'], result['log_var'], result['z']
                recon_loss, kl_loss = vae_loss(recon_batch, data, mu, log_var, mu_target=mu_target, reduction = 'sum')
                total_loss_sum += (recon_loss.item() + cfg.kl_weight * kl_loss.item())
                recon_loss_sum += recon_loss.item()
                kl_loss_sum += kl_loss.item()
                mean_var += torch.mean(torch.exp(log_var), dim=1).sum().item() #bug torch.sum(torch.exp(log_var), dim=1).sum().item() until 04/17 23:59

            if compare_model is not None:
                compare_result = results[1]
                compare_recon_batch, compare_mu, compare_log_var, compare_z = compare_result['recon_x'], compare_result['mu'], compare_result['log_var'], compare_result['z']
                mean_l2_compare += torch.mean((mu - compare_mu) ** 2, dim=1).sum().item()
                var_diff_compare += torch.mean(torch.exp(log_var - compare_log_var), dim=1).sum().item()
                mean_var_compare += torch.mean(torch.exp(compare_log_var), dim=1).sum().item()
    result = {}
    result['total_loss'] = total_loss_sum / len(data_loader.dataset)
    result['recon_loss'] = recon_loss_sum / len(data_loader.dataset)
    result['mean_var'] = mean_var / len(data_loader.dataset)
    if cfg.vq:
        result['codebook_loss'] = codebook_loss_sum / len(data_loader.dataset)
        result['commitment_loss'] = commitment_loss_sum / len(data_loader.dataset)
    else:
        result['kl_loss'] = kl_loss_sum / len(data_loader.dataset)

    if cfg.client_classifier:
        result['distance_from_client_embedding'] = client_class_celoss / len(data_loader.dataset)
        result['client_classification_accuracy'] = 100 * correct / len(data_loader.dataset)

    if cfg.use_classifier:
        result['accuracy'] = 100 * correct / len(data_loader.dataset)
    
    if compare_model is not None:
        result['mean_l2_compare'] = mean_l2_compare / len(data_loader.dataset)
        result['var_diff_compare'] = var_diff_compare / len(data_loader.dataset)
        result['mean_var_compare'] = mean_var_compare / len(data_loader.dataset)
    return result



# ==============================================
# ==   Calculate FID/IS Helper Function      ==
# ==============================================

def calculate_fid_is(cfg,
                     input_data_source: DataLoader,
                     num_samples: int = 10000, # Number of samples to use
                    ) -> Dict[str, float]:
    """
    Calculates FID and IS for generated samples using torch-fidelity.

    Args:
        cfg: Configuration object (needs data_dir).
        dataloader: DataLoader yielding batches of generated images (N, C, H, W), range [0, 1].
        num_samples_for_metric: How many samples to use for calculation.
        reference_stats_name: Name used to cache/load precomputed statistics of the
                              reference dataset (e.g., combined MNIST+Fashion train).

    Returns:
        A dictionary containing 'fid' and 'inception_score_mean'. Returns NaNs if calculation fails.
    """
    results = {'fid': np.nan, 'inception_score_mean': np.nan, 'inception_score_std': np.nan}

    # if dataloader is None or len(dataloader.dataset) == 0:
    #      print("Warning: Cannot calculate FID/IS on empty dataloader.")
    #      return results

    actual_samples_available = len(input_data_source)
    num_samples_for_metric = num_samples
    num_samples_to_use = min(num_samples_for_metric, actual_samples_available)

    if num_samples_to_use < 100: # Need a reasonable number of samples
        print(f"Warning: Not enough generated samples ({num_samples_to_use}) available for reliable FID/IS calculation. Skipping.")
        return results

    # print(f"Calculating FID/IS using {num_samples_to_use} generated samples...")

    # # --- Define Input Sampler for torch-fidelity ---
    # # This tells torch-fidelity how to get batches from our dataloader
    # class InputSampler:
    #     def __init__(self, loader, total_samples, device):
    #         self.loader_iter = iter(loader)
    #         self.total_samples = total_samples
    #         self.samples_yielded = 0
    #         self.device = device

    #     def __iter__(self):
    #         return self

    #     def __next__(self):
    #         if self.samples_yielded >= self.total_samples:
    #             raise StopIteration
    #         try:
    #             # Get data, ignore labels, move to device
    #             data, _ = next(self.loader_iter)
    #             data = data.to(self.device)
    #             # Ensure data is float and potentially replicate grayscale channel 3 times
    #             if data.shape[1] == 1: # Grayscale
    #                 data = data.repeat(1, 3, 1, 1) # Repeat channel to C=3
    #             data = data.float()

    #             batch_size = data.shape[0]
    #             samples_to_yield = min(batch_size, self.total_samples - self.samples_yielded)
    #             self.samples_yielded += samples_to_yield
    #             return data[:samples_to_yield] # Return only needed samples from batch
    #         except StopIteration:
    #              print("Warning: DataLoader exhausted before yielding required number of samples for FID/IS.")
    #              raise StopIteration # Propagate stop

    # # Create sampler instance
    # input_sampler = InputSampler(dataloader, num_samples_to_use, cfg.device)

    # if isinstance(input_data_source, DataLoader):
    #     print("Input is a DataLoader, extracting the underlying Dataset.")
    #     input_for_fidelity = input_data_source.dataset # <<< KEY CHANGE HERE
    #     # Note: If the DataLoader used subset indices, this dataset is the original full one.
    #     # torch-fidelity might need the subset Dataset if that's intended.
    #     # If using DatasetSplit, pass the DatasetSplit instance directly.
    #     # Check if input_data_source.dataset is the correct representation of the data you want to evaluate.
    # elif isinstance(input_data_source, Dataset):
    #     print("Input is a Dataset.")
    #     input_for_fidelity = input_data_source
    # elif isinstance(input_data_source, str):
    #     print("Input is a string (path or registered name).")
    #     input_for_fidelity = input_data_source
    # else:
    #     raise TypeError("Invalid input_data_source type for calculate_fid_is. "
    #                     "Expected DataLoader, Dataset, or str.")


    input_for_fidelity = FidelityInputDataset(input_data_source)
    #print whether input_for_fidelity is an instance of Dataset
    if isinstance(input_for_fidelity, Dataset):
        print("Input for fidelity is a Dataset.")
    else:
        print("Input for fidelity is not a Dataset.")
    # --- Define path for cached reference statistics ---
    # This speeds up subsequent calculations by not recomputing stats for the reference dataset
    stats_dir = os.path.join(cfg.data_dir)
    os.makedirs(stats_dir, exist_ok=True)
    reference_stats_path = os.path.join(stats_dir, ".fidelity_stats_cache.npz")
    #print(f"  Reference dataset stats path: {reference_stats_path}")

    # change reference_stats_path to reference_stats_name
    #reference_stats_name = reference_stats_path
    reference_image_dir = os.path.join(cfg.data_dir, "image_files")
    reference_stats_name = ".fidelity_stats_cache"
    metrics_dict = calculate_metrics(
        input1=input_for_fidelity, # Our generated samples
        input1_num_samples=num_samples_to_use,
        input2=reference_image_dir, # Name of reference dataset (or path to data)
        input2_cache_name=reference_stats_name,
        # Path where reference stats are cached (required if input2 is a name)
        input2_cache_dir=reference_stats_name,
        # Alternatively, provide path to reference data directly if cache doesn't exist:
        # input2='/path/to/your/reference/images/folder',
        cuda=False, #True,
        isc=True, # Calculate Inception Score
        fid=True, # Calculate FID
        verbose=False, # Set to True for detailed progress
        # Optional: Specify inception features dim (default 2048)
        # feature_layer_fid='2048',
        # feature_layer_isc='logits_unbiased',
    )
    # results['fid'] = metrics_dict.get('frechet_inception_distance', np.nan)
    # results['inception_score_mean'] = metrics_dict.get('inception_score_mean', np.nan)
    # results['inception_score_std'] = metrics_dict.get('inception_score_std', np.nan)
    # print(f"  FID: {results['fid']:.3f}, IS: {results['inception_score_mean']:.3f} +/- {results['inception_score_std']:.3f}")

    # except FileNotFoundError as e:
    #      print(f"Error calculating FID/IS: Reference dataset statistics not found at {reference_stats_path}.")
    #      print(f"  Please ensure the reference statistics exist or provide a path to the reference dataset folder using 'input2'.")
    #      print(f"  You might need to run calculate_metrics once with 'input2' pointing to your reference data folder to generate the cache.")
    #      print(f"  Example command to generate cache (run separately):")
    #      print(f"  python -m torch_fidelity --input2 /path/to/reference/data --input2-cache-name {reference_stats_name} --save-cpu-ram --cache-root {stats_dir}")
    #      # Example using dataloader to generate stats (might need adjustment)
    #      # print(f"  Alternatively, try generating stats from a reference dataloader:")
    #      # from utils.data import get_mnist_fashion_datasets # Needs this import
    #      # m_train, _, f_train, _ = get_mnist_fashion_datasets(cfg.data_dir)
    #      # ref_dataset = ConcatDataset([m_train, f_train])
    #      # ref_loader = DataLoader(ref_dataset, batch_size=64, num_workers=cfg.num_workers)
    #      # ref_sampler = InputSampler(ref_loader, len(ref_dataset), cfg.device)
    #      # try:
    #      #     calculate_metrics(input1=ref_sampler, input1_num_samples=len(ref_dataset), cuda=True, isc=False, fid=True, input1_cache_name=reference_stats_name, cache_root=stats_dir)
    #      #     print(f"  Generated reference stats cache: {reference_stats_path}")
    #      #     # Re-run the original FID/IS calculation now
    #      # except Exception as e_cache: print(f"  Failed to generate reference cache: {e_cache}")

    # except ValueError as e:
    #      # Handle potential errors like mismatched image sizes or formats
    #      print(f"Error calculating FID/IS (ValueError): {e}")
    #      print("  Ensure generated images have the expected size (e.g., 28x28) and format.")
    # except Exception as e:
    #     print(f"An unexpected error occurred during FID/IS calculation: {e}")

    return metrics_dict



import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF # Import functional transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import os
from torch_fidelity import calculate_metrics
from typing import Union, Dict
import gc

# Constants for FID
IMAGE_SIZE = 299
FID_MEAN = [0.485, 0.456, 0.406]
FID_STD = [0.229, 0.224, 0.225]

# --- FidelityInputDataset: Handles both PIL and Tensor inputs ---
class FidelityInputDataset(Dataset):
    def __init__(self, original_input):
        self.original_input = original_input
        # Determine if original_input is a Dataset or a Tensor/List of Tensors
        if isinstance(original_input, Dataset):
            self.is_dataset = True
            self.length = len(original_input)
        elif isinstance(original_input, (torch.Tensor, list)):
             if isinstance(original_input, list):
                 if len(original_input) == 0: self.length = 0
                 elif not all(torch.is_tensor(t) for t in original_input): raise TypeError("Input list must contain only tensors")
                 else: self.length = len(original_input)
             else: # Single Tensor (assume N, C, H, W or N, flat)
                 self.length = original_input.shape[0]
             self.is_dataset = False
        else:
            raise TypeError(f"Unsupported original_input type: {type(original_input)}")

        self._first_item_fetched = False # Flag to check the item shape once

    def __len__(self):
        return self.length

    def _get_raw_item(self, idx):
        """Gets the raw item (PIL or Tensor) before FID processing."""
        if self.is_dataset:
            item = self.original_input[idx]
            # Handle datasets returning (image, label) tuples
            if isinstance(item, (tuple, list)):
                raw_data = item[0]
            else:
                raw_data = item
        else: # Tensor or list of tensors f
            # Assumes original_input is (N, ...) or list[idx] is (...)
            raw_data = self.original_input[idx]
        return raw_data

    def __getitem__(self, idx):
        raw_data = self._get_raw_item(idx)

        try:
            # --- Apply Transformations based on initial type ---
            processed_tensor = None

            if isinstance(raw_data, Image.Image):
                # Path for PIL Image input
                # 1. Convert to RGB
                img_pil_rgb = raw_data.convert('RGB')
                # 2. Resize
                img_pil_resized = img_pil_rgb.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR) # PIL Resize
                # 3. Convert to Tensor [0, 1]
                img_tensor_float = TF.to_tensor(img_pil_resized) # Now (3, H, W), float [0,1]

            elif torch.is_tensor(raw_data):
                # Path for Tensor input
                img_tensor = raw_data
                # 0. Ensure float [0, 1] if input is uint8 [0, 255]
                if img_tensor.dtype == torch.uint8:
                     img_tensor = img_tensor.float() / 255.0

                # 1. Reshape if necessary (e.g., from flat 784 or (1, 28, 28))
                if img_tensor.ndim == 1 and img_tensor.shape[0] == 784: # Flat MNIST/Fashion
                    img_tensor = img_tensor.view(1, 28, 28)
                elif img_tensor.ndim == 3 and img_tensor.shape[0] != 1 and img_tensor.shape[0]!=3: # (H,W,C)? need (C,H,W)
                     if img_tensor.shape[2] in [1,3]: # Guessing H, W, C format
                         img_tensor = img_tensor.permute(2, 0, 1)
                     else: raise ValueError(f"Unsupported tensor shape: {img_tensor.shape}")
                elif img_tensor.ndim == 2: # (H, W) grayscale
                    img_tensor = img_tensor.unsqueeze(0) # Add channel dim -> (1, H, W)

                # Check shape after potential reshape
                if img_tensor.ndim != 3: raise ValueError(f"Tensor is not 3D after initial processing: {img_tensor.shape}")

                # 2. Ensure 3 Channels
                if img_tensor.shape[0] == 1:
                    img_tensor = img_tensor.repeat(3, 1, 1) # Repeat grayscale -> (3, H, W)
                elif img_tensor.shape[0] != 3:
                    raise ValueError(f"Tensor has unexpected channel size: {img_tensor.shape}")

                # 3. Resize using functional API (works on tensors)
                # Ensure input tensor is (C, H, W)
                img_tensor_float = TF.resize(img_tensor, [IMAGE_SIZE, IMAGE_SIZE], interpolation=TF.InterpolationMode.BILINEAR)

            else:
                raise TypeError(f"Unsupported data type at index {idx}: {type(raw_data)}")

            # --- Common Step: Convert float [0, 1] tensor to uint8 [0, 255] ---
            # Check if already normalized (heuristic: check min/max)
            # This avoids normalizing twice if input tensor was already normalized
            min_val, max_val = img_tensor_float.min(), img_tensor_float.max()
            if not (min_val >= -0.1 and max_val <= 1.1): # Heuristic for [0,1] range
                 print(f"Warning: Tensor at index {idx} seems outside [0,1] range ({min_val:.2f}, {max_val:.2f}) before uint8 conversion. Clamping.")
                 # Potentially apply inverse normalization if needed, or just clamp
                 # img_tensor_float = img_tensor_float * torch.tensor(FID_STD).view(3,1,1) + torch.tensor(FID_MEAN).view(3,1,1)

            processed_tensor = (img_tensor_float * 255).clamp(0, 255).to(torch.uint8)

            # Check shape only once for efficiency
            if not self._first_item_fetched:
                 if processed_tensor.shape != (3, IMAGE_SIZE, IMAGE_SIZE):
                      print(f"Warning: Processed tensor shape is {processed_tensor.shape}, expected {(3, IMAGE_SIZE, IMAGE_SIZE)}")
                 self._first_item_fetched = True


        except Exception as e:
            print(f"Error processing item in FidelityInputDataset at index {idx}: {e}")
            import traceback
            traceback.print_exc()
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8) # Return dummy

        # Final check
        if not (torch.is_tensor(processed_tensor) and processed_tensor.dtype == torch.uint8):
             raise TypeError(f"Final output for index {idx} is not torch.uint8. Type: {type(processed_tensor)}, Dtype: {processed_tensor.dtype if torch.is_tensor(processed_tensor) else 'N/A'}")

        return processed_tensor