import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import wandb
from dataclasses import dataclass,asdict, field
import argparse
from collections import OrderedDict
import cloudpickle
from scipy.stats import norm
import copy
import gc

@dataclass
class Config:
    num_epochs: field(default_factory=lambda: OrderedDict({"combined": 100}))
    #hidden_dims: list = [512, 256]
    batch_size: int = 128
    latent_dim: int = 2
    lr: float = 1e-3
    num_rounds: int = 25
    local_epochs: int = 4
    num_samples: int = 10
    save_dir: str = None
    project: str = "vae"
    name: str = "central_combined"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"



    @classmethod
    def central_combined(cls):
        return cls(name="central_combined", num_epochs={"combined": 100})
    
    @classmethod
    def central_mnist_combined(cls):
        return cls(name="central_mnist_combined", num_epochs={"mnist": 100, "combined": 100})
    
    @classmethod
    def central_combined_mnist(cls):
        return cls(name="central_combined_mnist", num_epochs={"combined": 100,"mnist": 100})

    @classmethod
    def central_mnist_fashion(cls):
        return cls(name="central_mnist_fashion", num_epochs={"mnist": 100, "fashion": 100})


    @classmethod
    def federated_rounds25_epochs4(cls):
        return cls(name="federated_rounds25_epochs4", num_epochs={}, num_rounds=25, local_epochs=4)

    @classmethod
    def federated_rounds50_epochs2(cls):
        return cls(name="federated_rounds50_epochs2", num_epochs={}, num_rounds=50, local_epochs=2)

    @classmethod
    def federated_rounds25_epochs8(cls):
        return cls(name="federated_rounds25_epochs8", num_epochs={}, num_rounds=25, local_epochs=8)

    @classmethod
    def federated_rounds200_epochs1(cls):
        return cls(name="federated_rounds200_epochs1", num_epochs={}, num_rounds=200, local_epochs=1)

    @classmethod
    def federated_rounds50_epochs2_legd(cls):
        return cls(name="federated_rounds50_epochs2_legd", num_epochs={}, num_rounds=50, local_epochs=2)

    @classmethod
    def federated_rounds25_epochs4_legd(cls):
        return cls(name="federated_rounds25_epochs4_legd", num_epochs={}, num_rounds=25, local_epochs=4)

    @classmethod
    def federated_rounds25_epochs8_legd(cls):
        return cls(name="federated_rounds25_epochs8_legd", num_epochs={}, num_rounds=25, local_epochs=8)

    @classmethod
    def federated_rounds200_epochs1_legd(cls):
        return cls(name="federated_rounds200_epochs1_legd", num_epochs={}, num_rounds=200, local_epochs=1)


    @classmethod
    def federated_rounds100_epochs2_geld(cls):
        return cls(name="federated_rounds100_epochs2_geld", num_epochs={}, num_rounds=100, local_epochs=2)
    
    @classmethod
    def federated_rounds50_epochs4_geld(cls):
        return cls(name="federated_rounds50_epochs4_geld", num_epochs={}, num_rounds=50, local_epochs=4)

    @classmethod
    def federated_rounds25_epochs8_geld(cls):
        return cls(name="federated_rounds25_epochs8_geld", num_epochs={}, num_rounds=25, local_epochs=8)

    @classmethod
    def federated_rounds200_epochs1_geld(cls):
        return cls(name="federated_rounds200_epochs1_geld", num_epochs={}, num_rounds=200, local_epochs=1)
    
    @classmethod
    def firstcentral_then_fed_encoder(cls):
        return cls(name="firstcentral_then_fed_encoder", num_epochs={"combined": 100}, num_rounds=200, local_epochs=1)

    @classmethod
    def firstcentral_then_fed_decoder(cls):
        return cls(name="firstcentral_then_fed_decoder", num_epochs={"combined": 100}, num_rounds=200, local_epochs=1) 

    @classmethod
    def firstcentral_then_fed_all(cls):
        return cls(name="firstcentral_then_fed_all", num_epochs={"combined": 100}, num_rounds=200, local_epochs=1)

    def asdict(self):
        return asdict(self)
    # def asdict(self):
    #     """Convert to dict, handling OrderedDict explicitly."""
    #     config_dict = vars(self)
    #     config_dict["num_epochs"] = dict(self.num_epochs)  # Convert OrderedDict to dict
    #     return config_dict

def get_config(exp_name="central_combined"):
    return getattr(Config, exp_name)()




# Set device (use GPU with specific number if available)
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transform
transform = transforms.ToTensor()

# Load training datasets
#
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

# Centralized: Combine datasets
combined_train = torch.utils.data.ConcatDataset([mnist_train, fashion_train])
centralized_loader = torch.utils.data.DataLoader(combined_train, batch_size=128, shuffle=True)

# Federated: legd loaders for each client
mnist_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)
fashion_loader = torch.utils.data.DataLoader(fashion_train, batch_size=64, shuffle=True)

# Test datasets (for test loss computation)
mnist_test = datasets.MNIST(
   "./data", train=False, download=True, transform=transform
)
fashion_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
mnist_test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=1000, shuffle=False)
fashion_test_loader = torch.utils.data.DataLoader(fashion_test, batch_size=1000, shuffle=False)

# Define VAE model
class VAE(nn.Module):
    def __init__(self, hidden_dims=[512, 256], latent_dim=2):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 4)  # 2 for mu, 2 for log_var
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], 784),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x.view(-1, 784))
        mu, log_var = h[:, :2], h[:, 2:]
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z)
        return recon_x, mu, log_var, z

    def encoder_forward(self, x):
        h = self.encoder(x.view(-1, 784))
        mu, log_var = h[:, :2], h[:, 2:]
        z = self.reparameterize(mu, log_var)
        return mu, log_var, z
    
    def decoder_forward(self, z):
        return self.decoder(z)

# VAE loss function
def vae_loss(recon_x, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

# Compute loss for a data loader
def compute_loss(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            recon_batch, mu, log_var, z = model(data)
            loss = vae_loss(recon_batch, data, mu, log_var)
            total_loss += loss.item()
    return total_loss / len(data_loader.dataset)


def compute_average_var(model, data_loader):
    model.eval()
    total_var = 0
    data_num = 0
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            mu, log_var, z = model.encoder_forward(data)
            total_var += ((torch.exp(log_var))).sum()
            data_num += len(data)
    return total_var / data_num

# Get latent representations
def get_latent_representations(model, data_loader):
    model.eval()
    mus, labels = [], []
    with torch.no_grad():
        for data, targets in data_loader:
            data = data.to(device)
            _, mu, _, _ = model(data)
            mus.append(mu.cpu().numpy())
            labels.append(targets.numpy())
    
    return np.concatenate(mus, axis=0), np.concatenate(labels, axis=0)

# Plot latent space
def plot_latent_space(model, mnist_loader, fashion_loader, title):
    mnist_mus, mnist_labels = get_latent_representations(model, mnist_loader)
    fashion_mus, fashion_labels = get_latent_representations(model, fashion_loader)
    
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
    def __init__(self, num_samples=10, save_dir=None):
        self.num_samples = num_samples
        # self.save_dir = save_dir
        # self.counter = 0

        # if self.save_dir and not os.path.exists(self.save_dir):
        #     os.makedirs(self.save_dir)

    def __call__(self, model, dataloader):
        model.eval()  # Set model to eval mode due to Dropout, BN, etc.
        with torch.no_grad():
            inputs = self._batch_random_samples(dataloader)
            outputs, mu, log_var, z = model(inputs)  # Forward pass

            # Prepare data for plotting
            input_images = self._reshape_to_image(inputs, numpy=True)
            recon_images = self._reshape_to_image(outputs, numpy=True)
            z_ = self._to_numpy(z)

            fig = self._plot_samples(input_images, recon_images, z_)

        model.train()  # Return to train mode
        return fig

    def _batch_random_samples(self, dataloader):
        """Helper function that retrieves `num_samles` from dataset and prepare them in batch for model """
        dataset = dataloader.dataset

        # Randomly sample one data sample per each class
        samples = []
        for i in range(10):
            idx = np.random.choice(np.where(dataset.targets == i)[0], size=1)
            #breakpoint()
            samples.append(dataset[idx[0]][0])

        # Create batch
        batch = torch.stack(samples)
        batch = batch.view(batch.size(0), -1)  # Flatten
        batch = batch.to(device)

        return batch

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


# visualize the latent space
def visualize_manifold(model, num_samples=20):
    x = norm.ppf(np.linspace(0.011, 0.99, num_samples))
    y = norm.ppf(np.linspace(0.011, 0.99, num_samples))
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

# Centralized training
def train_centralized(cfg):
    #wandb.init(project="vae", name="centralized_run")
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    #num_epochs = 100
    plot_callback = PlotCallback(num_samples=cfg.num_samples)
    epoch = 0
    for train_stage, stage_epochs in cfg.num_epochs.items():
        this_loader = centralized_loader if train_stage == "combined" else mnist_loader if train_stage == "mnist" else fashion_loader
        for stage_epoch in range(stage_epochs):
            model.train()
            train_loss = 0
            for data, _ in this_loader:
                data = data.to(device)
                optimizer.zero_grad()
                recon_batch, mu, log_var, z = model(data)
                loss = vae_loss(recon_batch, data, mu, log_var)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            
            train_loss_avg = train_loss / len(this_loader.dataset)
            mnist_test_loss_avg = (compute_loss(model, mnist_test_loader))
            fashion_test_loss_avg = (compute_loss(model, fashion_test_loader))
            
            latent_fig = plot_latent_space(model, mnist_test_loader, fashion_test_loader, f"Centralized Epoch {epoch+1}")
            mnist_recon_fig = plot_callback(model, mnist_loader)
            fashion_recon_fig = plot_callback(model, fashion_loader)
            manifold_fig = visualize_manifold(model)

            #log average var
            MNIST_var = compute_average_var(model, mnist_loader)
            Fashion_var = compute_average_var(model, fashion_loader)
            combined_var = compute_average_var(model, centralized_loader)


            wandb.log({
                #"epoch": epoch + 1,
                "central_train_loss": train_loss_avg,
                "central_mnist_test_loss": mnist_test_loss_avg,
                "central_fashion_test_loss": fashion_test_loss_avg,
                "central_latent_space": wandb.Image(latent_fig, caption="Centralized: MNIST (o), Fashion-MNIST (x)"),
                "mnist_reconstructions": wandb.Image(mnist_recon_fig, caption="MNIST Reconstructions"),
                "fashion_reconstructions": wandb.Image(fashion_recon_fig, caption="Fashion-MNIST Reconstructions"),
                "manifold": wandb.Image(manifold_fig, caption="Manifold"),
                "MNIST_var": MNIST_var,
                "Fashion_var": Fashion_var,
                "Combined_var": combined_var,
            },
            step=epoch + 1)
            plt.close(latent_fig)
            plt.close(mnist_recon_fig)
            plt.close(fashion_recon_fig)
            plt.close(manifold_fig)
            print(f"Centralized Epoch {epoch+1}, Train Loss: {train_loss_avg:.4f}, MNIST_Test Loss: {mnist_test_loss_avg:.4f}, Fashion_Test Loss: {fashion_test_loss_avg:.4f}")
            epoch += 1
    
    #andb.finish()
    return model


def analyze_model(cfg, model, title):
    model.eval()
    with torch.no_grad():
        plot_callback = PlotCallback(num_samples=cfg.num_samples)
        mnist_test_loss_avg = (compute_loss(model, mnist_test_loader))
        fashion_test_loss_avg = (compute_loss(model, fashion_test_loader))
        latent_fig = plot_latent_space(model, mnist_test_loader, fashion_test_loader, title)
        mnist_recon_fig = plot_callback(model, mnist_loader)
        fashion_recon_fig = plot_callback(model, fashion_loader)
        manifold_fig = visualize_manifold(model)
    model.train()

    return latent_fig, mnist_recon_fig, fashion_recon_fig, manifold_fig, mnist_test_loss_avg, fashion_test_loss_avg

# Client class for federated learning
class Client:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.model = VAE().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self, global_weights, local_epochs):
        self.model.load_state_dict(global_weights)
        self.model.train()
        for _ in range(local_epochs):
            for data, _ in self.data_loader:
                data = data.to(device)
                self.optimizer.zero_grad()
                recon_batch, mu, log_var, z = self.model(data)
                loss = vae_loss(recon_batch, data, mu, log_var)
                loss.backward()
                self.optimizer.step()
        return self.model.state_dict()

    
    def train_encoderonly(self, global_weights, local_epochs):
        self.model.load_state_dict(global_weights)
        self.model.train()
        #freeze decoder
        for param in self.model.decoder.parameters():
            param.requires_grad = False
        for _ in range(local_epochs):
            for data, _ in self.data_loader:
                data = data.to(device)
                self.optimizer.zero_grad()
                mu, log_var, z = self.model.encoder_forward(data)
                recon_batch = self.model.decoder_forward(z)
                loss = vae_loss(recon_batch, data, mu, log_var)
                loss.backward()
                self.optimizer.step()
        return self.model.state_dict()

    def train_decoderonly(self, global_weights, local_epochs):
        self.model.load_state_dict(global_weights)
        self.model.train()
        #freeze encoder
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        for _ in range(local_epochs):
            for data, _ in self.data_loader:
                data = data.to(device)
                self.optimizer.zero_grad()
                mu, log_var, z = self.model.encoder_forward(data)
                recon_batch = self.model.decoder_forward(z)
                loss = vae_loss(recon_batch, data, mu, log_var)
                loss.backward()
                self.optimizer.step()
        return self.model.state_dict()

    
    def train_legd(self, global_model, local_epochs):
        #freeze global_model parameter
        for param in global_model.parameters():
            param.requires_grad = False
        self.model.load_state_dict(global_model.state_dict())
        self.model.train()
        for _ in range(local_epochs):
            for data, _ in self.data_loader:
                data = data.to(device)
                self.optimizer.zero_grad()
                
                #process 1. local encoder + freezed global decoder
                mu, log_var, z = self.model.encoder_forward(data)
                recon_batch = global_model.decoder_forward(z)

                #process 2. detached local encoder output + local decoder. Make sure that encoder is not updated
                with torch.no_grad():
                    mu1, log_var1, z1 = self.model.encoder_forward(data)
                    mu1, log_var1, z1 = mu1.detach(), log_var1.detach(), z1.detach()
                recon_batch1 = self.model.decoder_forward(z1)

                loss = vae_loss(recon_batch, data, mu, log_var) + vae_loss(recon_batch1, data, mu1, log_var1)
                loss.backward()
                self.optimizer.step()
        return self.model.state_dict()

    def train_geld(self, global_model, local_epochs):
            #freeze global_model parameter
            for param in global_model.parameters():
                param.requires_grad = False
            self.model.load_state_dict(global_model.state_dict())
            self.model.train()
            for _ in range(local_epochs):
                for data, _ in self.data_loader:
                    #create freezed local decoder
                    freezed_decoder = copy.deepcopy(self.model.decoder)
                    for param in freezed_decoder.parameters():
                        param.requires_grad = False

                    data = data.to(device)
                    self.optimizer.zero_grad()
                    
                    #process 1. freezed global encoder + local decoder
                    with torch.no_grad():
                        mu1, log_var1, z1 = global_model.encoder_forward(data)
                    recon_batch1 = self.model.decoder_forward(z1)

                    #process 2. local encoder output + freezed local decoder. Make sure that decoder is not updated
                    mu, log_var, z = self.model.encoder_forward(data)
                    #forward through local decoder. But do not update the parameters
                    recon_batch = freezed_decoder(z)


                    loss = vae_loss(recon_batch, data, mu, log_var) + vae_loss(recon_batch1, data, mu1, log_var1)
                    loss.backward()
                    self.optimizer.step()
            return self.model.state_dict()


# Server class for federated learning
class Server:
    def __init__(self, global_model=None):
        if global_model is None:
            self.global_model = VAE().to(device)
        else:
            self.global_model = global_model

    def aggregate(self, client_weights):
        avg_weights = {key: sum(w[key] for w in client_weights) / len(client_weights) for key in client_weights[0].keys()}
        self.global_model.load_state_dict(avg_weights)

# Federated training
def train_federated(cfg, server: Server=None, wandb_initstep=0):
    if server is None:
        server = Server()
    #wandb.init(project="vae", name="federated_run")
    MNISTClient = Client(mnist_loader)
    FashionClient = Client(fashion_loader)
    num_rounds = cfg.num_rounds
    local_epochs = cfg.local_epochs
    
    for round_num in range(num_rounds):
        wandb_results = {}
        if "legd" in cfg.name:
            client_weights = [
                MNISTClient.train_legd(server.global_model, local_epochs),
                FashionClient.train_legd(server.global_model, local_epochs)
            ]
        
        elif "geld" in cfg.name:
            client_weights = [
                MNISTClient.train_geld(server.global_model, local_epochs),
                FashionClient.train_geld(server.global_model, local_epochs)
            ]
        elif "firstcentral_then_fed_encoder" in cfg.name:
            client_weights = [
                MNISTClient.train_encoderonly(server.global_model.state_dict(), local_epochs),
                FashionClient.train_decoderonly(server.global_model.state_dict(), local_epochs)
            ]
        elif "firstcentral_then_fed_decoder" in cfg.name:
            client_weights = [
                MNISTClient.train_decoderonly(server.global_model.state_dict(), local_epochs),
                FashionClient.train_encoderonly(server.global_model.state_dict(), local_epochs)
            ]
        elif "firstcentral_then_fed_all" in cfg.name:
            client_weights = [
                MNISTClient.train(server.global_model.state_dict(), local_epochs),
                FashionClient.train(server.global_model.state_dict(), local_epochs)
            ]
        else:
            client_weights = [
                MNISTClient.train(server.global_model.state_dict(), local_epochs),
                FashionClient.train(server.global_model.state_dict(), local_epochs)
            ]
        server.aggregate(client_weights)
        
        ## Eval && analysis
        figures_to_close = []
        

        # Analyzie MNISTClient.model
        MNISTClient_latent_fig, MNISTClient_mnist_recon_fig, MNISTClient_fashion_recon_fig, MNISTClient_manifold_fig, MNISTClient_mnist_test_loss_avg, MNISTClient_fashion_test_loss_avg = analyze_model(cfg, MNISTClient.model, f"Client 1 Round {round_num+1}")

        # Analyzie FashionClient.model
        FashionClient_latent_fig, FashionClient_mnist_recon_fig, FashionClient_fashion_recon_fig, FashionClient_manifold_fig, FashionClient_mnist_test_loss_avg, FashionClient_fashion_test_loss_avg = analyze_model(cfg, FashionClient.model, f"Client 2 Round {round_num+1}")

        wandb_results.update({
            "MNISTClient_latent_space": wandb.Image(MNISTClient_latent_fig, caption="Client 1: MNIST (o), Fashion-MNIST (x)"),
            "MNISTClient_mnist_reconstructions": wandb.Image(MNISTClient_mnist_recon_fig, caption="Client 1 MNIST Reconstructions"),
            "MNISTClient_fashion_reconstructions": wandb.Image(MNISTClient_fashion_recon_fig, caption="Client 1 Fashion-MNIST Reconstructions"),
            "MNISTClient_manifold": wandb.Image(MNISTClient_manifold_fig, caption="Client 1 Manifold"),
            "MNISTClient_mnist_test_loss": MNISTClient_mnist_test_loss_avg,
            "MNISTClient_fashion_test_loss": MNISTClient_fashion_test_loss_avg,
            "FashionClient_latent_space": wandb.Image(FashionClient_latent_fig, caption="Client 2: MNIST (o), Fashion-MNIST (x)"),
            "FashionClient_mnist_reconstructions": wandb.Image(FashionClient_mnist_recon_fig, caption="Client 2 MNIST Reconstructions"),
            "FashionClient_fashion_reconstructions": wandb.Image(FashionClient_fashion_recon_fig, caption="Client 2 Fashion-MNIST Reconstructions"),
            "FashionClient_manifold": wandb.Image(FashionClient_manifold_fig, caption="Client 2 Manifold"),
            "FashionClient_mnist_test_loss": FashionClient_mnist_test_loss_avg,
            "FashionClient_fashion_test_loss": FashionClient_fashion_test_loss_avg,
        })
        
        figures_to_close.extend([MNISTClient_latent_fig, MNISTClient_mnist_recon_fig, MNISTClient_fashion_recon_fig, MNISTClient_manifold_fig, FashionClient_latent_fig, FashionClient_mnist_recon_fig, FashionClient_fashion_recon_fig, FashionClient_manifold_fig])

        mnist_train_loss_avg = compute_loss(MNISTClient.model, mnist_loader)
        fashion_train_loss_avg = compute_loss(FashionClient.model, fashion_loader)

        wandb_results.update({
            "MNIST_train_loss": mnist_train_loss_avg,
            "Fashion_train_loss": fashion_train_loss_avg,
        })
        
        # Analyzie server.global_model
        server_latent_fig, server_mnist_recon_fig, server_fashion_recon_fig, server_manifold_fig, server_mnist_test_loss_avg, server_fashion_test_loss_avg = analyze_model(cfg, server.global_model, f"Server Round {round_num+1}")

        wandb_results.update({
            "MNIST_test_loss": server_mnist_test_loss_avg,
            "Fashion_test_loss": server_fashion_test_loss_avg,
            "server_latent_space": wandb.Image(server_latent_fig, caption="Server: MNIST (o), Fashion-MNIST (x)"),
            "server_mnist_reconstructions": wandb.Image(server_mnist_recon_fig, caption="Server MNIST Reconstructions"),
            "server_fashion_reconstructions": wandb.Image(server_fashion_recon_fig, caption="Server Fashion-MNIST Reconstructions"),
            "server_manifold": wandb.Image(server_manifold_fig, caption="Server Manifold"),
        })
        figures_to_close.extend([server_latent_fig, server_mnist_recon_fig, server_fashion_recon_fig, server_manifold_fig])


        wandb.log(wandb_results, step=round_num + 1)
        print(f"Federated Round {round_num+1}/{num_rounds}, MNIST Train Loss: {mnist_train_loss_avg:.4f}, Fashion Train Loss: {fashion_train_loss_avg:.4f}, MNIST Test Loss: {server_mnist_test_loss_avg:.4f}, Fashion Test Loss: {server_fashion_test_loss_avg:.4f}")

        # Close figures
        for fig in figures_to_close:
            plt.close(fig)
        gc.collect()
    
    #wandb.finish()
    return server.global_model


# First, train the full vae encoder and decoder with centralized data. Then, train the encoder from scratch using fedavg while keeping the decoder fixed.
# Finally, train the decoder from scratch using fedavg while keeping the encoder fixed.
# This is the upperbound case for training the parameters with fedavg, because we have the ideal decoder or encoder in each case.
def train_upperbound(cfg):
    central_model = train_centralized(cfg)
    fed_model = train_federated(cfg, server=Server(global_model=central_model))
    return fed_model
# Run training
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="central")
    args = parser.parse_args()
    cfg = get_config(exp_name=args.name)
    wandb.init(entity="FedRL-SNU", project=cfg.project, name=cfg.name)

    # wandb config
    wandb.config.update(cfg.asdict())
    global device 
    device = cfg.device

    if 'firstcentral_then_fed' in cfg.name:
        upperbound_model = train_upperbound(cfg)

    elif 'central' in cfg.name:
        centralized_model = train_centralized(cfg)
    elif "federated" in cfg.name:
        federated_model = train_federated(cfg)
    wandb.finish()

