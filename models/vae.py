import torch
import torch.nn as nn
from utils.data import idx2onehot

__all__ = ['VAE']

# Define VAE model
class VAE(nn.Module):
    def __init__(self, hidden_dims=[512, 256], latent_dim=2, conditional=False, num_classes=20, batch_norm=False):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.conditional = conditional
        self.num_classes = num_classes
        self.batch_norm = batch_norm
            
        self.encoder = nn.Sequential(
            nn.Linear(784, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]) if self.batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]) if self.batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], latent_dim*2)  # 2 for mu, 2 for log_var
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]) if self.batch_norm else nn.Identity(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]) if self.batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], 784),
            nn.Sigmoid()
        )
        if self.conditional:
            self.encoder[0] = nn.Linear(784 + num_classes, hidden_dims[0])
            self.decoder[0] = nn.Linear(latent_dim + num_classes, hidden_dims[1])

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c=None):
        
        x = x.view(-1, 784)

        if self.conditional:
            c = idx2onehot(c, n=self.num_classes)
            x = torch.cat((x, c), dim=-1)

        h = self.encoder(x)
        mu, log_var = h[:, :self.latent_dim], h[:, self.latent_dim:]
        z = self.reparameterize(mu, log_var)

        if self.conditional:
            z = torch.cat((z, c), dim=-1)

        recon_x = self.decoder(z)
        return recon_x, mu, log_var, z

    def encoder_forward(self, x, c=None):
        x = x.view(-1, 784)

        if self.conditional:
            c = idx2onehot(c, n=self.num_classes)
            x = torch.cat((x, c), dim=-1)

        h = self.encoder(x)
        mu, log_var = h[:, :self.latent_dim], h[:, self.latent_dim:]
        z = self.reparameterize(mu, log_var)
        return mu, log_var, z
    
    def decoder_forward(self, z, c=None):
        if self.conditional:
            z = torch.cat((z, c), dim=-1)

        return self.decoder(z)