import torch
import torchvision
import torch.nn as nn
import numpy as np

__all__ = ['VQVAE']

# Code from https://github.com/explainingai-code/VQVAE-Pytorch/blob/main/run_simple_vqvae.py



class VQVAE(nn.Module):
    def __init__(self, *args, **kwargs):
        super(VQVAE, self).__init__()
        self.cfg = kwargs['cfg']
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        
        self.pre_quant_conv = nn.Conv2d(4, 2, kernel_size=1)
        self.embedding = nn.Embedding(num_embeddings=self.cfg.num_embeddings, embedding_dim=2)
        self.post_quant_conv = nn.Conv2d(2, 4, kernel_size=1)
        
        
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
        # B, C, H, W
        encoded_output = self.encoder(x)
        quant_input = self.pre_quant_conv(encoded_output)
        
        ## Quantization
        B, C, H, W = quant_input.shape
        quant_input = quant_input.permute(0, 2, 3, 1)
        quant_input = quant_input.reshape((quant_input.size(0), -1, quant_input.size(-1)))
        
        # Compute pairwise distances
        dist = torch.cdist(quant_input, self.embedding.weight[None, :].repeat((quant_input.size(0), 1, 1)))
        
        # Find index of nearest embedding
        min_encoding_indices = torch.argmin(dist, dim=-1)
        
        # Select the embedding weights
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1))
        quant_input = quant_input.reshape((-1, quant_input.size(-1)))
        
        # Compute losses

        commitment_loss = ((quant_out.detach() - quant_input)**2)
        codebook_loss = ((quant_out - quant_input.detach())**2)
        #quantize_losses = codebook_loss + self.beta*commitment_loss
        
        # Ensure straight through gradient
        quant_out = quant_input + (quant_out - quant_input).detach()
        if self.cfg.vq_noquantize:
            quant_out = quant_input
            commitment_loss = commitment_loss.detach()
            codebook_loss = codebook_loss.detach()
        
        # Reshaping back to original input shape
        quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
        min_encoding_indices = min_encoding_indices.reshape((-1, quant_out.size(-2), quant_out.size(-1)))
        
        
        ## Decoder part
        decoder_input = self.post_quant_conv(quant_out)
        output = self.decoder(decoder_input)
        output = output.view(-1, 784)
        return output, codebook_loss, commitment_loss