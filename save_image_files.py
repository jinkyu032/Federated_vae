import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm # Progress bar

# --- Configuration ---
DATA_DIR = './data'  # MNIST/FashionMNIST 데이터셋이 저장될/있는 경로
OUTPUT_BASE_DIR = './data/image_files' # 이미지 파일을 저장할 기본 경로
FORCE_RESAVE = False # True로 설정하면 기존 파일이 있어도 덮어씀

# --- Function to load and save a dataset ---
def save_dataset_as_images(dataset_name, dataset_class, train, output_dir):
    """Loads a torchvision dataset and saves images as PNG files."""
    print(f"\nProcessing {dataset_name} {'Train' if train else 'Test'} set...")

    # --- Load Dataset ---
    try:
        # Load as PIL Images directly
        print(f"  Loading dataset from {DATA_DIR}...")
        # Handle potential MNIST mirror issues
        if dataset_class == datasets.MNIST:
            try:
                dataset = dataset_class(root=DATA_DIR, train=train, download=True, transform=None)
            except Exception as e:
                print(f"  Default MNIST download failed ({e}), trying alternative mirror...")
                new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
                datasets.MNIST.resources = [
                    ('/'.join([new_mirror, url.split('/')[-1]]), md5)
                    for url, md5 in datasets.MNIST.resources
                ]
                dataset = dataset_class(root=DATA_DIR, train=train, download=True, transform=None)
        else:
             dataset = dataset_class(root=DATA_DIR, train=train, download=True, transform=None)
        print(f"  Loaded {len(dataset)} images.")
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        return

    # --- Prepare Output Directory ---
    os.makedirs(output_dir, exist_ok=True)
    num_existing_files = len([name for name in os.listdir(output_dir) if name.endswith('.png')])

    if not FORCE_RESAVE and num_existing_files >= len(dataset):
        print(f"  Skipping saving: Found {num_existing_files} images, matching dataset size {len(dataset)}.")
        return
    elif not FORCE_RESAVE and num_existing_files > 0:
         print(f"  Warning: Found {num_existing_files} existing images, but dataset size is {len(dataset)}. Re-saving might be needed if incomplete.")
         # Decide if you want to continue saving anyway or stop
         # continue

    print(f"  Saving images to {output_dir}...")
    # Use tqdm for progress bar
    for i, (image, label) in enumerate(tqdm(dataset, desc=f"Saving {dataset_name} {'Train' if train else 'Test'}")):
        # Image is already a PIL Image
        prefix = 'train' if train else 'test'
        # Include label in filename for potential future use (optional)
        image_filename = f'{dataset_name.lower()}_{prefix}_{i:05d}_label{label}.png'
        image_path = os.path.join(output_dir, image_filename)

        try:
            image.save(image_path)
        except Exception as e:
            print(f"\nError saving image {i} to {image_path}: {e}")
            # Decide how to handle: skip, retry, stop?
            # continue

    print(f"  Finished saving {dataset_name} {'Train' if train else 'Test'} images.")


# --- Main Script Logic ---
if __name__ == "__main__":
    print("Starting image saving process...")

    datasets_to_process = [
        ("MNIST", datasets.MNIST, True),       # MNIST Train
        ("MNIST", datasets.MNIST, False),      # MNIST Test
        ("FashionMNIST", datasets.FashionMNIST, True), # FashionMNIST Train
        ("FashionMNIST", datasets.FashionMNIST, False) # FashionMNIST Test
    ]

    for name, dset_class, is_train in datasets_to_process:
        subset_name = 'train' if is_train else 'test'
        output_path = os.path.join(OUTPUT_BASE_DIR)#, f"{name}_{subset_name}")
        save_dataset_as_images(name, dset_class, is_train, output_path)

    print("\nImage saving process complete.")
    print(f"Images saved in subdirectories under: {OUTPUT_BASE_DIR}")