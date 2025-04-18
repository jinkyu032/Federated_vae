import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
from utils.logging_utils import AverageMeter
from utils.losses import vae_loss,celoss  # Assuming you have this loss function
from torch.nn import CrossEntropyLoss
import random
import copy

__all__ = ['VAEClientClassifierClient']

from clients.client_withclassifier import VAEClassifierClient

# Client class for federated learning with VAE and Classifier
class VAEClientClassifierClient(VAEClassifierClient):
    def __init__(self, cfg: Dict, model: nn.Module, data_loader: Optional[DataLoader] = None, vae_mu_target: Optional[int] = None, *args, **kwargs):
        super(VAEClientClassifierClient, self).__init__(cfg=cfg, model=model, data_loader=data_loader, vae_mu_target=vae_mu_target, *args, **kwargs)
        self.client_idx = kwargs.get('client_idx', None)
        self.subset_id = kwargs.get('subset_id', None)  # Assuming subset_id is passed in kwargs
        self.cfg = cfg
        
        if self.client_idx is None:
            raise ValueError("client_idx must be provided in the config.")

        self.embedding_loss_weight = cfg.embedding_loss_weight
        self.num_of_clients = cfg.num_clients
        self.use_consistency_loss = getattr(cfg, 'use_consistency_loss', False) # Default to False
        self.consistency_loss_weight = getattr(cfg, 'consistency_loss_weight', 0.1) # Example weight
        self.num_consistency_samples = getattr(cfg, 'num_consistency_samples', 1) # Number of other clients to sample per iter
        self.gen_vae_loss_weight = getattr(cfg, 'gen_vae_loss_weight', 0.1) # Default or from cfg


        if self.use_consistency_loss and self.num_of_clients is None:
             raise ValueError("cfg.num_clients must be provided when use_consistency_loss is True.")
        if self.use_consistency_loss and self.num_of_clients <= 1:
             print("Warning: Consistency loss enabled, but num_clients <= 1. Loss will not be calculated.")
             self.use_consistency_loss = False # Disable if only one client

    def train(self, local_epochs, global_rounds = 0):
        current_lr = self.cfg.lr * (self.cfg.lr_decay ** global_rounds if self.cfg.lr_decay > 0 else 1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=current_lr)

        loss_meter = AverageMeter('Loss', ':.2f')
        recon_loss_meter = AverageMeter('Recon Loss', ':.2f')
        kl_loss_meter = AverageMeter('KL Loss', ':.2f')
        embedding_loss_meter = AverageMeter('Classifier Loss', ':.2f')
        consistency_loss_meter = AverageMeter('Consist Loss', ':.2f')
        gen_vae_loss_meter = AverageMeter('Gen VAE Loss', ':.4f') 
        assert(self.client_idx is not None)
        self.global_model = copy.deepcopy(self.model)
        #Freeze global model parameters
        for name, param in self.global_model.named_parameters():
            param.requires_grad = False

            

        self.model.train()
        for _ in range(local_epochs):
            for data, target in self.data_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                loss = 0.0
                if self.use_consistency_loss and self.num_of_clients > 1:
                    with torch.no_grad():

                        combined_client_idxs_for_decoder = torch.ones(data.size(0), dtype=torch.long, device=self.device) * self.subset_id if self.cfg.condition_subset else self.client_idx
                        consistency_loss = torch.tensor(0.0, device=self.device) # Initialize loss for this iteration
                        # 1. Sample other client IDs
                        all_client_indices = list(range(self.num_of_clients))
                        all_client_indices.remove(self.client_idx) # Exclude self
                        
                        # Ensure we don't sample more than available other clients
                        #num_to_sample = min(self.num_consistency_samples, len(all_client_indices))
                        if self.num_consistency_samples > 0:
                            sampled_other_idxs = torch.tensor([random.sample(all_client_indices, 1)[0] for _ in range(self.num_consistency_samples)]).to(self.device)

                        z_prior = torch.randn(self.num_consistency_samples, self.cfg.latent_dim, device=self.device)
                        if self.cfg.gen_global_model:
                            generated_recon_x = self.global_model.decoder_forward(z_prior, sampled_other_idxs).detach()
                        else:
                            generated_recon_x = self.model.decoder_forward(z_prior, sampled_other_idxs).detach()
                        #breakpoint() # Debugging point
                        generated_recon_x = generated_recon_x.reshape(-1, *data.shape[1:])

                        concat_data = torch.cat((data, generated_recon_x), dim=0)
                        concat_target = torch.cat((combined_client_idxs_for_decoder, sampled_other_idxs), dim=0)
                    result_combined = self.model(concat_data, concat_target, return_classfier_output=True)
                    result = {}
                    result_gen = {}
                    for key in result_combined.keys():
                        result[key] = result_combined[key][:data.size(0)]
                        result_gen[key] = result_combined[key][data.size(0):]
                    recon_gen_batch, mu_gen, log_var_gen, z_gen, prob_class_output_gen = result_gen['recon_x'], result_gen['mu'], result_gen['log_var'], result_gen['z'], result_gen['prob_class_output']
                    # The GT label is the client index we used for generation
                    recon_gen_loss, kl_gen_loss = self.vae_loss(recon_gen_batch, generated_recon_x, mu_gen, log_var_gen, mu_target=self.vae_mu_target, reduction = self.cfg.reduction)
                    current_consistency_loss = celoss(prob_class_output_gen, sampled_other_idxs, reduction=self.cfg.reduction)
                    gen_vae_loss = recon_gen_loss + self.kl_weight * kl_gen_loss
                    loss_gen = gen_vae_loss * self.gen_vae_loss_weight + self.consistency_loss_weight * current_consistency_loss
                    loss += loss_gen

                else:
                    concat_data = data
                    concat_target = target
                    result = self.model(data, self.client_idx, return_classfier_output=True)


                #recon_batch, mu, log_var, z, class_output = self.model(data, self.client_embedding ,return_classfier_output=True)
                
                recon_batch, mu, log_var, z, prob_class_output = result['recon_x'], result['mu'], result['log_var'], result['z'], result['prob_class_output']

                # VAE Loss
                recon_loss, kl_loss = self.vae_loss(recon_batch, data, mu, log_var, mu_target=self.vae_mu_target, reduction = self.cfg.reduction)
                vae_total_loss = recon_loss + self.kl_weight * kl_loss


                # Classifier Loss
                client_target = torch.ones(data.size(0),dtype=torch.long).to(self.cfg.device) * self.client_idx
                embedding_loss = celoss(prob_class_output, client_target, reduction=self.cfg.reduction)
                

                # Total Loss
                loss_real = vae_total_loss + self.embedding_loss_weight * embedding_loss  # Joint training, adjust weight as needed
                loss += loss_real 
                



                
                loss.backward()
                self.optimizer.step()

                loss_meter.update(loss.item(), data.size(0))
                recon_loss_meter.update(recon_loss.item(), data.size(0))
                kl_loss_meter.update(kl_loss.item(), data.size(0))
                embedding_loss_meter.update(embedding_loss.item(), data.size(0))
                if self.use_consistency_loss and self.num_of_clients > 1:
                    consistency_loss_meter.update(current_consistency_loss.item(), data.size(0))
                    gen_vae_loss_meter.update(gen_vae_loss.item(), data.size(0))

        print(f"Training Loss: {loss_meter.avg:.2f}, Recon Loss: {recon_loss_meter.avg:.2f}, KL Loss: {kl_loss_meter.avg:.2f}, Classifier Loss: {embedding_loss_meter.avg:.2f}")
        loss_dict = {
            "train_loss": loss_meter.avg,
            "train_recon_loss": recon_loss_meter.avg,
            "train_kl_loss": kl_loss_meter.avg,
            "train_embedding_loss": embedding_loss_meter.avg
        }
        if self.use_consistency_loss and self.num_of_clients > 1:
            print(f"Consistency Loss: {consistency_loss_meter.avg:.2f}, Gen VAE Loss: {gen_vae_loss_meter.avg:.2f}")
            loss_dict["train_consistency_loss"] = consistency_loss_meter.avg
            loss_dict["train_gen_vae_loss"] = gen_vae_loss_meter.avg
        return self.model.state_dict(), loss_dict

    def update_model(self, global_weights):
        self.model.load_state_dict(global_weights)