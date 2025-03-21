import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from torch.utils.data import DataLoader
from typing import Dict, Union
from omegaconf import DictConfig

__all__ = ['plot_latent_space', 'plot_latent_per_client', 'plot_latent_per_client', 'plot_recontruction_from_noise']

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
    def __init__(self, cfg: Union[Dict, DictConfig], num_samples=10, save_dir=None, device=None):
        self.num_samples = num_samples
        self.device = device
        self.cfg = cfg
        if isinstance(cfg, DictConfig):
            self.num_total_classes = cfg.model.num_total_classes    
        else:
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
    
def plot_recontruction_from_noise(cfg: Union[Dict, DictConfig], model, num_samples=10, device=None, mu=0, std=1):
    if isinstance(cfg, DictConfig):
        cfg = cfg.model
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
def visualize_manifold(model, num_samples=20, device=None, offset = (0, 0)):
    x = norm.ppf(np.linspace(0.011, 0.99, num_samples)) + offset[0]
    y = norm.ppf(np.linspace(0.011, 0.99, num_samples)) + offset[1]
    inputs = [(i, j) for i in x for j in y]
    inputs_t = torch.tensor(inputs, dtype=torch.float32).to(device)  
    model.eval()  
    with torch.no_grad():
        outputs = model.decoder(inputs_t)
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
