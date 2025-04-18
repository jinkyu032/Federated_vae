# save_statistics.py

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset # Added Dataset for type hint
import numpy as np
import os
from typing import Tuple # Added for type hint

# Attempt to import from local utils if available, otherwise define here
try:
    from utils import get_mnist_fashion_datasets # Assuming your utils file is structured like this
    print("Imported get_mnist_fashion_datasets from utils.")
except ImportError:
    print("Could not import from utils, defining get_mnist_fashion_datasets locally.")
    # Define the function directly if not importable
    def get_mnist_fashion_datasets(data_dir: str = './data', download: bool = True) -> Tuple[Dataset, Dataset, Dataset, Dataset]:
        """Loads MNIST and FashionMNIST datasets, adjusting FashionMNIST targets."""
        transform = transforms.Compose([ transforms.ToTensor(), ]) # Base transform for loading

        # Handle potential mirror issues for MNIST
        try:
            # Try default first
            mnist_trainset = datasets.MNIST(data_dir, train=True, download=download, transform=transform)
            mnist_testset = datasets.MNIST(data_dir, train=False, download=download, transform=transform)
        except Exception as e:
            print(f"Default MNIST download failed ({e}), trying alternative mirror...")
            new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
            datasets.MNIST.resources = [
                ('/'.join([new_mirror, url.split('/')[-1]]), md5)
                for url, md5 in datasets.MNIST.resources
            ]
            mnist_trainset = datasets.MNIST(data_dir, train=True, download=download, transform=transform)
            mnist_testset = datasets.MNIST(data_dir, train=False, download=download, transform=transform)

        fashion_trainset = datasets.FashionMNIST(data_dir, train=True, download=download, transform=transform)
        fashion_testset = datasets.FashionMNIST(data_dir, train=False, download=download, transform=transform)

        # Adjust FashionMNIST targets to be 10-19
        # Check for 'targets' attribute first
        if hasattr(fashion_trainset, 'targets'):
            # Ensure targets are tensors for arithmetic
            if not isinstance(fashion_trainset.targets, torch.Tensor):
                fashion_trainset.targets = torch.tensor(fashion_trainset.targets)
            if not isinstance(fashion_testset.targets, torch.Tensor):
                fashion_testset.targets = torch.tensor(fashion_testset.targets)

            fashion_trainset.targets = fashion_trainset.targets + 10
            fashion_testset.targets = fashion_testset.targets + 10
        # Fallback to 'labels' if 'targets' doesn't exist
        elif hasattr(fashion_trainset, 'labels'):
             # Ensure labels are tensors for arithmetic
            if not isinstance(fashion_trainset.labels, torch.Tensor):
                fashion_trainset.labels = torch.tensor(fashion_trainset.labels)
            if not isinstance(fashion_testset.labels, torch.Tensor):
                fashion_testset.labels = torch.tensor(fashion_testset.labels)

            fashion_trainset.labels = fashion_trainset.labels + 10
            fashion_testset.labels = fashion_testset.labels + 10
        else:
            print("Warning: Could not find 'targets' or 'labels' attribute in FashionMNIST dataset.")

        return mnist_trainset, mnist_testset, fashion_trainset, fashion_testset

# Import necessary functions from pytorch_fid
try:
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import calculate_activation_statistics
except ImportError:
    print("\nError: pytorch-fid library not found.")
    print("Please install it using: pip install pytorch-fid")
    exit()

# --- Configuration ---
DATA_DIR = './data'       # Directory where datasets are stored/downloaded
STATS_FILENAME = './mnist_fashion_real_stats.npz' # Output NPZ file name
BATCH_SIZE = 64           # Batch size for processing
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DIMS = 2048               # Inception feature dimensionality
NUM_WORKERS = 4           # DataLoader workers
IMAGE_SIZE = 299          # Target image size for InceptionV3

# --- Main Script Logic ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # --- 1. Load and Combine Datasets ---
    print("Loading MNIST and FashionMNIST datasets...")
    try:
        mnist_train, _, fashion_train, _ = get_mnist_fashion_datasets(data_dir=DATA_DIR, download=True)
        # Note: Fashion MNIST targets are already adjusted (10-19) by the function
        combined_train_dataset = ConcatDataset([mnist_train, fashion_train])
        print(f"Combined training dataset size: {len(combined_train_dataset)}")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        exit()

    # --- 2. Define FID Transforms ---
    # Transforms applied AFTER loading the PIL image from dataset
    fid_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Lambda(lambda x: x.convert('RGB')), # Ensure RGB
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
    ])

    # --- 3. Create DataLoader for Combined Dataset with FID Transforms ---
    # We need a wrapper Dataset that applies the FID transform on top of the base dataset
    class FIDTransformedDataset(Dataset):
        def __init__(self, original_dataset, transform):
            self.original_dataset = original_dataset
            self.transform = transform

        def __len__(self):
            return len(self.original_dataset)

        def __getitem__(self, idx):
            # Get the original PIL image and label
            image, label = self.original_dataset[idx]
            # Apply the FID-specific transforms
            transformed_image = self.transform(image)
            # Return only the transformed image, as FID calculation doesn't need labels
            return transformed_image

    print("Creating DataLoader with FID transforms...")
    fid_dataset = FIDTransformedDataset(combined_train_dataset, fid_transform)
    dataloader = DataLoader(fid_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False, # No shuffling needed
                            drop_last=False,
                            num_workers=NUM_WORKERS,
                            pin_memory=True if DEVICE.startswith('cuda') else False) # Use pin_memory with GPU

    # --- 4. Load Inception-v3 Model ---
    print("Loading Inception-v3 model...")
    try:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[DIMS]
        model = InceptionV3([block_idx]).to(DEVICE)
        model.eval()
        print("Inception-v3 model loaded.")
    except Exception as e:
        print(f"Error loading Inception model: {e}")
        exit()

    # --- 5. Calculate Activation Statistics ---
    print(f"\nCalculating activation statistics (dim={DIMS})...")
    try:
        mu, sigma = calculate_activation_statistics(
                                    # Pass the DataLoader directly
                                    data_loader=dataloader,
                                    model=model,
                                    batch_size=BATCH_SIZE, # Still useful for the function's internal batching
                                    dims=DIMS,
                                    device=DEVICE,
                                    # num_workers arg in calculate_activation_statistics might be ignored
                                    # if dataloader is passed, but doesn't hurt to include
                                    num_workers=NUM_WORKERS
                                    )
        print("Statistics calculation complete.")

        # --- 6. Save Statistics ---
        print(f"Saving statistics to {STATS_FILENAME}...")
        # Check if directory exists, create if not
        output_dir = os.path.dirname(STATS_FILENAME)
        if output_dir and not os.path.exists(output_dir):
             os.makedirs(output_dir)
        np.savez(STATS_FILENAME, mu=mu, sigma=sigma)
        print(f"Successfully saved statistics (mu shape: {mu.shape}, sigma shape: {sigma.shape}).")

    except Exception as e:
        print(f"\nAn error occurred during statistics calculation or saving:")
        print(e)
        import traceback
        traceback.print_exc()

    print("\nScript finished.")