import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import numpy as np
import os
import json
from typing import List, Dict, Optional, Tuple, Union
from contextlib import contextmanager
from collections import defaultdict


__all__ = ['DatasetSplit', 'get_dataloaders_federated', 'get_mnist_fashion_datasets','get_dataloaders', 'idx2onehot', 'split_data_federated', 'get_targets', 'get_mnist_fashion_datasets', 'DatasetSplit']


def get_dataloaders(cfg):
    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor()
        ])

    # Load training datasets
    new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
    datasets.MNIST.resources = [
    ('/'.join([new_mirror, url.split('/')[-1]]), md5)
    for url, md5 in datasets.MNIST.resources
    ]
    mnist_train = datasets.MNIST(
    "./data", train=True, download=True, transform=transform
    )
    #mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    fashion_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    #if cfg.conditional:
    fashion_train.targets = fashion_train.targets + 10

    # Federated: legd loaders for each client
    mnist_loader = torch.utils.data.DataLoader(mnist_train, batch_size=cfg.batch_size, shuffle=True)
    fashion_loader = torch.utils.data.DataLoader(fashion_train, batch_size=cfg.batch_size, shuffle=True)

    # Test datasets (for test loss computation)
    mnist_test = datasets.MNIST(
    "./data", train=False, download=False, transform=transform
    )
    fashion_test = datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
    #if cfg.conditional:
    fashion_test.targets = fashion_test.targets + 10
    mnist_test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=cfg.eval_batch_size, shuffle=False)
    fashion_test_loader = torch.utils.data.DataLoader(fashion_test, batch_size=cfg.eval_batch_size, shuffle=False)

    return mnist_loader, mnist_test_loader, fashion_loader, fashion_test_loader



def idx2onehot(idx, n):

    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot


# ==============================================
# ==      Helper Classes and Functions        ==
# ==============================================

@contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

# Use the exact DatasetSplit class provided by the user
class DatasetSplit(torch.utils.data.Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs, *args, **kwargs):
        
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs] # Ensure integer indices
        self.class_dict = {}
        self.client_id = kwargs.get('client_id', None)
        self.subset_id = kwargs.get('subset_id', None)
        for idx in self.idxs:
            image, label = self.dataset[idx]
            if torch.is_tensor(label): label_str = str(label.item())
            else: label_str = str(label)
            if label_str in self.class_dict: self.class_dict[label_str] += 1
            else: self.class_dict[label_str] = 1


    def __len__(self): return len(self.idxs)
    def __getitem__(self, item):
        if item >= len(self.idxs): raise IndexError(f"Index {item} out of bounds")
        return self.dataset[self.idxs[item]]
    @property
    def num_classes(self): return len(self.class_dict.keys())
    @property
    def class_ids(self): return self.class_dict.keys()


def get_mnist_fashion_datasets(data_dir: str = './data', download: bool = True) -> Tuple[Dataset, Dataset, Dataset, Dataset]:
    # (Implementation from previous answer)
    transform = transforms.Compose([ transforms.ToTensor(), ])
    mnist_trainset = datasets.MNIST(data_dir, train=True, download=download, transform=transform)
    mnist_testset = datasets.MNIST(data_dir, train=False, download=download, transform=transform)
    fashion_trainset = datasets.FashionMNIST(data_dir, train=True, download=download, transform=transform)
    fashion_testset = datasets.FashionMNIST(data_dir, train=False, download=download, transform=transform)
    if hasattr(fashion_trainset, 'targets'):
        if isinstance(fashion_trainset.targets, tuple): fashion_trainset.targets = list(fashion_trainset.targets)
        if isinstance(fashion_testset.targets, tuple): fashion_testset.targets = list(fashion_testset.targets)
        fashion_trainset.targets = torch.as_tensor(fashion_trainset.targets) + 10
        fashion_testset.targets = torch.as_tensor(fashion_testset.targets) + 10
    elif hasattr(fashion_trainset, 'labels'):
        if isinstance(fashion_trainset.labels, tuple): fashion_trainset.labels = list(fashion_trainset.labels)
        if isinstance(fashion_testset.labels, tuple): fashion_testset.labels = list(fashion_testset.labels)
        fashion_trainset.labels = torch.as_tensor(fashion_trainset.labels) + 10
        fashion_testset.labels = torch.as_tensor(fashion_testset.labels) + 10
    else: print("Warning: Could not find standard 'targets' or 'labels' attribute.")
    return mnist_trainset, mnist_testset, fashion_trainset, fashion_testset

###############################################################################################################
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, random_split
from sklearn.model_selection import train_test_split
from torchvision.datasets import EuroSAT

class ColoredFashionMNIST(datasets.FashionMNIST):
    def __init__(self, root, train, download, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.color_map = {
                            10: (1.0, 0.0, 0.0),   # T‑셔츠/상의 → 빨강
                            11: (0.0, 1.0, 0.0),   # 바지       → 초록
                            12: (0.0, 0.0, 1.0),   # 풀오버     → 파랑
                            13: (1.0, 1.0, 0.0),   # 드레스     → 노랑
                            14: (1.0, 0.0, 1.0),   # 코트       → 마젠타
                            15: (0.0, 1.0, 1.0),   # 샌들       → 시안
                            16: (0.5, 0.5, 0.0),   # 셔츠       → 올리브
                            17: (0.5, 0.0, 0.5),   # 스니커즈   → 보라
                            18: (0.0, 0.5, 0.5),   # 가방       → 청록
                            19: (0.33, 0.66, 0.33) # 앵클 부츠  → 연두
                        }

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        gray = TF.to_tensor(img)  # shape = (1, H, W)
        v = gray.squeeze(0)       # shape = (H, W)
        color = torch.tensor(self.color_map[label], dtype=torch.float32) \
                        .view(3, 1, 1)
        colored = v.unsqueeze(0).repeat(3, 1, 1) * color  # shape = (3, H, W)
        return colored, label

class EuroSATSplit(EuroSAT):
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform=None,
        download: bool = False,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        super().__init__(root=root, transform=transform, download=download)

        if hasattr(self, 'targets'):
            labels = np.array(self.targets)
        else:
            labels = np.array([label for _, label in self.samples])

        # 2) stratified split
        indices = np.arange(len(labels))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )

        if split == 'train':
            self.split_idx = train_idx
        elif split == 'test':
            self.split_idx = test_idx
        else:
            raise ValueError("split keyword must be 'train' or 'test'")

    def __len__(self):
        return len(self.split_idx)

    def __getitem__(self, idx):
        real_idx = self.split_idx[idx]
        return super().__getitem__(real_idx)

def get_eurosat_fashion_datasets(data_dir: str = './data', download: bool = True) -> Tuple[Dataset, Dataset, Dataset, Dataset]:
    transform = transforms.Compose([
        transforms.Resize((28, 28)),     
        transforms.ToTensor(),            
    ])
    fashion_trainset = ColoredFashionMNIST(data_dir, train=True, download=True, transform=None)
    fashion_testset = ColoredFashionMNIST(data_dir, train=False, download=True, transform=None)
    # Eurosat
    eurosat_train = EuroSATSplit(data_dir, split='train', transform=transform, download=True, test_size=0.2, random_state=42)
    eurosat_test = EuroSATSplit(data_dir, split='test', transform=transform, download=False,   test_size=0.2,random_state=42)
    
    if hasattr(fashion_trainset, 'targets'):
        if isinstance(fashion_trainset.targets, tuple): fashion_trainset.targets = list(fashion_trainset.targets)
        if isinstance(fashion_testset.targets, tuple): fashion_testset.targets = list(fashion_testset.targets)
        fashion_trainset.targets = torch.as_tensor(fashion_trainset.targets) + 10
        fashion_testset.targets = torch.as_tensor(fashion_testset.targets) + 10
    elif hasattr(fashion_trainset, 'labels'):
        if isinstance(fashion_trainset.labels, tuple): fashion_trainset.labels = list(fashion_trainset.labels)
        if isinstance(fashion_testset.labels, tuple): fashion_testset.labels = list(fashion_testset.labels)
        fashion_trainset.labels = torch.as_tensor(fashion_trainset.labels) + 10
        fashion_testset.labels = torch.as_tensor(fashion_testset.labels) + 10
    else: print("Warning: Could not find standard 'targets' or 'labels' attribute.")
    return eurosat_train, eurosat_test, fashion_trainset, fashion_testset
####################################################################################################################


def _get_targets(dataset: Dataset) -> np.ndarray:
    # (Implementation from previous answer)
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
        if isinstance(targets, list): return np.array(targets)
        elif isinstance(targets, torch.Tensor): return targets.cpu().numpy()
        else: return np.array(targets)
    elif hasattr(dataset, 'labels'):
        labels = dataset.labels
        if isinstance(labels, list): return np.array(labels)
        elif isinstance(labels, torch.Tensor): return labels.cpu().numpy()
        else: return np.array(labels)
    elif isinstance(dataset, ConcatDataset):
         return torch.cat([torch.as_tensor(_get_targets(d)) for d in dataset.datasets]).numpy()
    elif isinstance(dataset, Subset):
         original_targets = _get_targets(dataset.dataset)
         indices_np = np.array(dataset.indices)
         valid_indices = indices_np[(indices_np >= 0) & (indices_np < len(original_targets))]
         if len(valid_indices) != len(indices_np): print("Warning: Subset indices out of bounds.")
         return original_targets[valid_indices]
    else:
        #print(f"Warning: Iterating to extract targets for {type(dataset)} (slow).")
        try:
            num_to_check = len(dataset)
            labels = [d[1] for d in [dataset[i] for i in range(num_to_check)]]
            return np.array([int(l.item()) if torch.is_tensor(l) else int(l) if isinstance(l, (int, float)) else -1 for l in labels])
        except Exception as e: raise TypeError(f"Could not extract labels/targets from {type(dataset)}. Error: {e}")


# split_data_federated remains the same as the last version provided (2-stage logic)
def split_data_federated(cfg, train_dataset: Dataset) -> Dict[int, np.ndarray]:

    num_clients = cfg.num_clients
    split_mode = cfg.split_mode.lower()
    num_samples = len(train_dataset)
    all_indices = np.arange(num_samples)

    # --- Configuration Validation ---
    num_subsets = getattr(cfg, 'num_subsets', None)
    if num_subsets is None or num_subsets <= 0:
        raise ValueError("cfg.num_subsets must be set > 0 for this splitting logic.")
    if num_clients % num_subsets != 0:
        print(f"Warning: num_clients ({num_clients}) not perfectly divisible by num_subsets ({num_subsets}). Client distribution per subset might be uneven.")

    # --- Get Labels and Define Subsets ---
    targets = _get_targets(train_dataset)
    num_classes_total = len(np.unique(targets)) # Should be 20
    print(f"Total samples: {num_samples}, Total unique classes: {num_classes_total}")
    assert(num_classes_total % num_subsets == 0)
    classes_per_subset = int(np.ceil(num_classes_total / num_subsets))

    subset_class_map: Dict[int, List[int]] = defaultdict(list)
    subset_indices_map: Dict[int, np.ndarray] = defaultdict(lambda: np.array([], dtype=np.int64))

    for k in range(num_classes_total):
        subset_id = k // classes_per_subset
        subset_class_map[subset_id].append(k)
        indices_k = np.where(targets == k)[0]
        subset_indices_map[subset_id] = np.concatenate((subset_indices_map[subset_id], indices_k))

    print(f"Defined {num_subsets} class subsets.")

    # --- Stage 1: Assign Clients to Subsets ---
    clients_per_subset = num_clients // num_subsets
    client_subset_assignment: Dict[int, int] = {} # {client_id: subset_id}
    all_client_ids = list(range(num_clients))
    #np.random.shuffle(all_client_ids)

    assigned_count = 0
    for subset_id in range(num_subsets):
        num_to_assign = clients_per_subset + (1 if subset_id < num_clients % num_subsets else 0)
        assigned_clients = all_client_ids[assigned_count : assigned_count + num_to_assign]
        for client_id in assigned_clients:
            client_subset_assignment[client_id] = subset_id
        assigned_count += num_to_assign
    print(f"Assigned {num_clients} clients to {num_subsets} subsets.")

    # --- Stage 2: Distribute Data Within Each Subset ---
    client_final_indices_dict: Dict[int, np.ndarray] = {i: np.array([], dtype=np.int64) for i in range(num_clients)}

    for subset_id in range(num_subsets):
        subset_indices = subset_indices_map[subset_id]
        if len(subset_indices) == 0: continue
        subset_targets = targets[subset_indices]
        subset_num_samples = len(subset_indices)
        subset_classes = subset_class_map[subset_id]
        subset_num_classes = len(subset_classes)

        clients_in_subset = [cid for cid, sid in client_subset_assignment.items() if sid == subset_id]
        num_clients_in_subset = len(clients_in_subset)

        if num_clients_in_subset == 0: continue
        print(f"\nDistributing data for Subset {subset_id} (Classes: {subset_classes}) among {num_clients_in_subset} clients using mode: {split_mode}")


        num_data_per_client_in_subset = subset_num_samples // num_clients_in_subset
        if subset_num_samples % num_clients_in_subset != 0:
             print(f"  Warning: Subset {subset_id} samples ({subset_num_samples}) not perfectly divisible by clients ({num_clients_in_subset}). Sample counts may vary slightly.")
        
        if split_mode == 'iid':
            np.random.shuffle(subset_indices)
            split_points = np.array_split(subset_indices, num_clients_in_subset)
            for i, client_id in enumerate(clients_in_subset):
                client_final_indices_dict[client_id] = split_points[i]

        elif split_mode =='skew':
            shards_per_client = getattr(cfg, 'num_classes_per_client', 1) # Reinterpret as shards/client
            print(f"  Applying Shard-based distribution within Subset {subset_id}: {shards_per_client} shards per client.")

            # Sort indices by label within the subset
            subset_indices_sorted_by_label = subset_indices[np.argsort(subset_targets)]

            # Create shards
            num_total_shards_in_subset = num_clients_in_subset * shards_per_client
            if subset_num_samples < num_total_shards_in_subset:
                print(f"  Warning: Not enough samples ({subset_num_samples}) for {num_total_shards_in_subset} shards. Creating {subset_num_samples} shards.")
                num_total_shards_in_subset = subset_num_samples # Each sample becomes a shard

            # Split into shards of roughly equal size
            actual_shards = np.array_split(subset_indices_sorted_by_label, num_total_shards_in_subset)
            num_actual_shards = len(actual_shards)
            idx_shard_available = list(range(num_actual_shards))

            # Assign shards to clients
            for client_id in clients_in_subset:
                num_shards_to_assign = min(shards_per_client, len(idx_shard_available))
                if num_shards_to_assign == 0: continue

                chosen_shard_indices_relative = np.random.choice(len(idx_shard_available), num_shards_to_assign, replace=False)
                chosen_shard_indices_absolute = [idx_shard_available[i] for i in chosen_shard_indices_relative]

                new_available_list = []
                chosen_set = set(chosen_shard_indices_absolute)
                for idx in idx_shard_available:
                    if idx not in chosen_set: new_available_list.append(idx)
                idx_shard_available = new_available_list

                client_idxs_list = [actual_shards[shard_idx] for shard_idx in chosen_shard_indices_absolute]

                if client_idxs_list:
                    final_indices = np.concatenate(client_idxs_list)
                    #np.random.shuffle(final_indices)
                    client_final_indices_dict[client_id] = final_indices.astype(np.int64)

        elif split_mode == 'dirichlet':
            alpha = cfg.dirichlet_alpha
            if alpha is None or alpha <= 0: raise ValueError("cfg.dirichlet_alpha > 0 needed.")
            print(f"  Applying Dirichlet Balanced (Multinomial Sampling) within Subset {subset_id} with alpha={alpha}.")

            # --- Using adapted cifar_dirichlet_balanced logic ---
            with temp_seed(getattr(cfg, 'seed', 0) + subset_id):
                # Need subset targets as torch tensor for multinomial weights
                y_train_subset_tensor = torch.from_numpy(subset_targets).long()
                N_subset = subset_num_samples
                # Map subset classes to 0..K'-1 if needed, but weights work on original indices
                K_subset = len(subset_classes) # Number of unique classes *in this subset*

                # Map subset_indices (global index) to relative index (0 to N_subset-1)
                subset_global_to_relative_idx = {global_idx: i for i, global_idx in enumerate(subset_indices)}

                # Initialize index lists for clients in this subset
                subset_client_indices_assigned: Dict[int, List[int]] = {cid: [] for cid in clients_in_subset}
                assigned_ids_mask_subset = torch.zeros(N_subset, dtype=torch.bool) # Track assignment *within subset*

                for i, client_id in enumerate(clients_in_subset):
                    weights_subset = torch.zeros(N_subset, dtype=torch.float32)
                    proportions = np.random.dirichlet(np.repeat(alpha, K_subset)) # Proportions for classes *in this subset*

                    # Assign weights based on class proportions to *unassigned* samples in subset
                    for k_idx, k_original in enumerate(subset_classes):
                         # Find relative indices within subset for class k
                         relative_indices_k = np.where(subset_targets == k_original)[0]
                         # Filter by those not yet assigned within the subset
                         mask_k_unassigned = ~assigned_ids_mask_subset[relative_indices_k]
                         valid_relative_indices_k = relative_indices_k[mask_k_unassigned]

                         if len(valid_relative_indices_k) > 0:
                             # Assign proportion to these available relative indices
                             weights_subset[valid_relative_indices_k] = torch.tensor(proportions[k_idx], dtype=torch.float32)

                    # Determine number of samples for this client
                    num_assigned_total = assigned_ids_mask_subset.sum().item()
                    num_remaining_total = N_subset - num_assigned_total
                    clients_remaining = num_clients_in_subset - i
                    # Calculate fair share, ensuring last client gets remainder
                    num_to_sample = num_data_per_client_in_subset
                    if i == num_clients_in_subset - 1: # Last client
                         num_to_sample = num_remaining_total
                    else: # Not the last client
                         # Adjust ideal count based on remaining samples/clients
                         num_to_sample = min(num_data_per_client_in_subset, num_remaining_total // clients_remaining if clients_remaining > 0 else num_remaining_total)

                    num_to_sample = max(0, num_to_sample) # Ensure non-negative


                    if weights_subset.sum() > 1e-9 and num_to_sample > 0:
                        # Get available relative indices and their weights
                        available_relative_indices = torch.where(weights_subset > 0)[0]
                        available_weights = weights_subset[available_relative_indices]

                        num_to_sample = min(num_to_sample, len(available_relative_indices)) # Can't sample more than available

                        if len(available_relative_indices) > 0 and num_to_sample > 0:
                             # Sample relative indices using multinomial
                             probabilities = available_weights #/ available_weights.sum()
                             chosen_relative_indices = torch.multinomial(probabilities, num_to_sample, replacement=False)
                             # Map back to global indices
                             chosen_global_indices = subset_indices[available_relative_indices[chosen_relative_indices]].tolist()

                             subset_client_indices_assigned[client_id].extend(chosen_global_indices)
                             assigned_ids_mask_subset[available_relative_indices[chosen_relative_indices]] = True # Mark as assigned within subset
                        else:
                             print(f"  Warning: Client {client_id}, Subset {subset_id} - No samples drawn via multinomial.")
                    else:
                         print(f"  Warning: Client {client_id}, Subset {subset_id} - No samples assigned (weights sum={weights_subset.sum()}, to_sample={num_to_sample})")

                # Assign final indices from multinomial sampling
                for client_id in clients_in_subset:
                    final_indices = np.array(subset_client_indices_assigned[client_id], dtype=np.int64)
                    np.random.shuffle(final_indices)
                    client_final_indices_dict[client_id] = final_indices

        else:
            breakpoint()
            raise ValueError(f"Unknown split_mode: {split_mode}.")

    if cfg.get('log_client_distribution', True):
        print("\n--- Final Client Data Distribution (After Stage 2) ---")
        total_assigned_final = 0
        for client_id in range(num_clients):
            indices = client_final_indices_dict.get(client_id, np.array([]))
            count = len(indices)
            total_assigned_final += count
            if count > 0:
                 client_labels = targets[indices]
                 unique, counts = np.unique(client_labels, return_counts=True)
                 dist_str = ", ".join([f"{cls}:{cnt}" for cls, cnt in sorted(zip(unique, counts))])
                 subset_assigned = client_subset_assignment.get(client_id, 'N/A')
                 print(f"Client {client_id} (Subset {subset_assigned}): {count} samples. Final Dist: {dist_str}")
            else:
                 print(f"Client {client_id} (Subset {client_subset_assignment.get(client_id, 'N/A')}): 0 samples.")
        print(f"Total assigned samples after Stage 2: {total_assigned_final}/{num_samples}")
        print("------------------------------------------------------\n")

    return client_final_indices_dict, client_subset_assignment


# ==============================================
# ==      Main DataLoader Function            ==
# ==============================================
# get_dataloaders_federated remains the same
def get_dataloaders_federated(cfg) -> Tuple[Dict[int, DataLoader], DataLoader, DataLoader, DataLoader]:
    # (Implementation from previous answer, including file caching logic)
    print("Loading and preparing datasets for federated learning...")
    mnist_trainset, mnist_testset, fashion_trainset, fashion_testset = get_mnist_fashion_datasets(data_dir=cfg.data_dir, download=True)
    print("Base datasets loaded.")
    combined_train_dataset = ConcatDataset([mnist_trainset, fashion_trainset])
    print(f"Combined training dataset size: {len(combined_train_dataset)}")

    split_dir = cfg.client_path 
    os.makedirs(split_dir, exist_ok=True)
    filename_parts = [ f"{cfg.split_mode}", f"{cfg.num_clients}cli", f"{cfg.num_subsets}subsets" ]
    if cfg.split_mode == 'noniid-class' or cfg.split_mode == 'skew':
        filename_parts.append(f"numcls{cfg.num_classes_per_client}") # Using nshpc as it uses shards now
    elif cfg.split_mode == 'dirichlet':
        filename_parts.append(f"alpha{cfg.dirichlet_alpha}")
    filename_parts.append(f"seed{getattr(cfg, 'seed', 0)}")
    idx_filename = "_".join(filename_parts) + ".json"
    idx_filepath = os.path.join(split_dir, idx_filename)
    print(f"Index file path: {idx_filepath}")

    client_indices: Dict[int, np.ndarray] = {}
    client_subset_assignment: Dict[int, int] = {}
    force_resplit = getattr(cfg, 'force_resplit', False)

    if os.path.exists(idx_filepath) and not force_resplit:
        print("Loading cached client indices...")
        try:
            with open(idx_filepath, 'r') as f:
                # Load the combined dictionary
                cached_data = json.load(f)
                indices_loaded = cached_data.get('indices', {})
                assignment_loaded = cached_data.get('assignment', {})

                # Convert loaded lists back to numpy arrays and int keys
                client_indices = {int(k): np.array(v, dtype=np.int64) for k, v in indices_loaded.items()}
                # Convert assignment keys to int
                client_subset_assignment = {int(k): v for k, v in assignment_loaded.items()}

                print(f"Successfully loaded indices for {len(client_indices)} clients.")
                print(f"Successfully loaded assignments for {len(client_subset_assignment)} clients.")

                # Sanity checks
                if len(client_indices) != cfg.num_clients or len(client_subset_assignment) != cfg.num_clients:
                    print(f"Warning: Loaded data mismatch (Indices: {len(client_indices)}, Assignment: {len(client_subset_assignment)}) vs cfg.num_clients ({cfg.num_clients}). Forcing resplit.")
                    client_indices = {}
                    client_subset_assignment = {} # Reset both
                else: print("Loaded indices successfully.")
                
        except Exception as e: print(f"Error loading index file: {e}. Regenerating.")
    else:
        if force_resplit: print("Forcing index regeneration...")
        else: print("Cached index file not found. Generating new split...")

    if not client_indices:
        client_indices, client_subset_assignment = split_data_federated(cfg, combined_train_dataset)
        print(f"Saving generated indices to: {idx_filepath}")
        try:
            # Prepare data for saving (convert numpy arrays to lists)
            indices_to_save = {k: v.tolist() for k, v in client_indices.items()}
            # Assignments are already int:int, but ensure keys are strings for JSON
            assignment_to_save = {str(k): v for k, v in client_subset_assignment.items()}
            data_to_save = {
                'indices': indices_to_save,
                'assignment': assignment_to_save
            }
            with open(idx_filepath, 'w') as f:
                json.dump(data_to_save, f, indent=4)
            print("Indices and assignments saved successfully.")
        except Exception as e: print(f"Error saving index file: {e}")
    #return client_indices

    print("Creating client training dataloaders...")
    train_loaders = {}
    

    for client_id, idxs in client_indices.items():
        client_id_int = int(client_id)
        if len(idxs) == 0: continue
        #client_class == client subset id
        split_dataset = DatasetSplit(combined_train_dataset, idxs, client_id=client_id_int, subset_id=client_subset_assignment[client_id_int])
        train_loaders[client_id_int] = DataLoader(split_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=getattr(cfg, 'num_workers', 4), pin_memory=True, drop_last=False)

    print("Creating test dataloaders...")
    combined_test_dataset = ConcatDataset([mnist_testset, fashion_testset])
    test_loader = DataLoader( combined_test_dataset, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=getattr(cfg, 'num_workers', 4), pin_memory=True)
    mnist_test_loader = DataLoader(mnist_testset, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=getattr(cfg, 'num_workers', 4))
    fashion_test_loader = DataLoader(fashion_testset, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=getattr(cfg, 'num_workers', 4))
    print("DataLoaders created successfully.")

    return train_loaders, test_loader, mnist_test_loader, fashion_test_loader, client_subset_assignment, filename_parts
