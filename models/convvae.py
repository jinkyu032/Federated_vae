import torch
import torchvision
import torch.nn as nn
import numpy as np

__all__ = ['CONVVAE']

# Code from https://github.com/explainingai-code/VQVAE-Pytorch/blob/main/run_simple_vqvae.py



class CONVVAE(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CONVVAE, self).__init__()
        self.cfg = kwargs['cfg']
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        
        if self.cfg.fully_conv:
            self.pre_quant_conv = nn.Conv2d(4, 4, kernel_size=1) # two for mu, two for log_var
            #self.embedding = nn.Embedding(num_embeddings=self.cfg.num_embeddings, embedding_dim=2)
            self.post_quant_conv = nn.Conv2d(2, 4, kernel_size=1)
        
        else:
            self.pre_quant_conv = nn.Linear(4*7*7, 4*7*7)
            self.post_quant_conv = nn.Linear(2*7*7, 4*7*7)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            #nn.Tanh(),
            #Note that the original code uses Tanh, but here we use Sigmoid, because we need to use binary cross entropy loss.
            nn.Sigmoid(),
        )
        
        
    def forward(self, x, c=None):
        mu, log_var, z = self.encoder_forward(x, c)
        recon_x = self.decoder_forward(z, c)
        #print(recon_x.shape)
        #return recon_x, mu, log_var, z
        result_dict = {
            "recon_x": recon_x,
            "mu": mu,
            "log_var": log_var,
            "z": z
        }
        return result_dict


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


    def encoder_forward(self, x, c=None):
        encoded_output = self.encoder(x)
        if not self.cfg.fully_conv:
            encoded_output = encoded_output.view(-1, 4*7*7)
        quant_input = self.pre_quant_conv(encoded_output)
        len_feature = quant_input.size(1)
        if self.cfg.fully_conv:
            mu, log_var = quant_input[:, :len_feature//2, :, :], quant_input[:, len_feature//2:, :, :]
        else:
            mu, log_var = quant_input[:, :len_feature//2], quant_input[:, len_feature//2:]
        z = self.reparameterize(mu, log_var)
        #flatten all
        mu = mu.view(mu.size(0), -1)
        log_var = log_var.view(log_var.size(0), -1)
        z = z.view(z.size(0), -1)
        return mu, log_var, z

    def decoder_forward(self, z, c=None):
        if self.cfg.fully_conv:
            z = z.view(z.size(0), 2, 7, 7)
        decoder_input = self.post_quant_conv(z)
        if not self.cfg.fully_conv:
            decoder_input = decoder_input.view(-1, 4, 7, 7)
        output = self.decoder(decoder_input)
        output = output.view(output.size(0), -1)
        return output
        