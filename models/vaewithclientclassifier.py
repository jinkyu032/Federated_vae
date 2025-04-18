import torch
import torch.nn as nn
from utils.data import idx2onehot
from .cosineclassifier import cosine_classifier

__all__ = ['VAEWithClientClassifier']

class VAEWithClientClassifier(nn.Module):
    def __init__(self, hidden_dims=[512, 256], latent_dim=2, num_classes=20, batch_norm=False, cfg = None):
        super(VAEWithClientClassifier, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.batch_norm = batch_norm
        self.cfg = cfg
        self.embedding_dim = cfg.embedding_dim
        self.num_subsets = cfg.num_subsets

        #num_of_clients = cfg.num_clients
        # embedding_dim = cfg.embedding_dim
        # client_embeddings = torch.randn(num_of_clients, embedding_dim).to(cfg.device)
        # client_embeddings.requires_grad = False
        # client_embeddings = F.normalize(client_embeddings, dim=1)

        self.num_of_clients = cfg.num_clients
        self.client_embeddings = nn.Parameter(torch.randn(self.num_subsets if cfg.condition_subset else self.num_of_clients, self.embedding_dim), requires_grad=False)
        self.client_embeddings.requires_grad = False
        #self.client_embeddings = torch.tensor([[1.0,0.0],[0.0,1.0]]).to(cfg.device)
        # self.client_idx = None
        # self.client_embedding = None

        self.cosine_classifier = cosine_classifier(self.embedding_dim, self.num_of_clients)


        # Unconditional Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]) if self.batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]) if self.batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], latent_dim * 2 + self.embedding_dim)  # 2 for mu, 2 for log_var
        )
        self.client_class_encoder = nn.Sequential(
            nn.Linear(784, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]) if self.batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]) if self.batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], self.embedding_dim)
        )

        # Decoder (takes concatenated latent and classifier output)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + self.embedding_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]) if self.batch_norm else nn.Identity(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]) if self.batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], 784),
            nn.Sigmoid()
        )

    def get_client_embedding(self, client_idx):
        with torch.no_grad():
            if client_idx is not None:
                return self.client_embeddings[client_idx]
            else:
                return None


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, client_idx = None,return_classfier_output=False):
        self.client_embeddings.requires_grad = False
        x = x.view(-1, 784)

        # Encoder
        h = self.encoder(x)
        mu, log_var = h[:, :self.latent_dim], h[:, self.latent_dim:2* self.latent_dim]
        z = self.reparameterize(mu, log_var)

        # Classifier
        #class_output = h[:,2* self.latent_dim:]
        if self.cfg.separate_client_encoder:
            class_output = self.client_class_encoder(x)
        else:
            class_output = h[:,2* self.latent_dim:]

        prob_class_output = self.cosine_classifier(class_output)

        #detach
        #prob_class_output = self.cosine_classifier(class_output.detach().clone())

        client_idxs = torch.argmax(prob_class_output, dim=1)
        if client_idx == None:
            client_embedding = self.client_embeddings[client_idxs]
        else: 
            if type(client_idx) == int:
                gt_client_idxs = torch.ones(z.size(0),dtype=torch.long).to(self.cfg.device) * client_idx
            else:
                # when client_idx is a tensor
                gt_client_idxs = client_idx
            
            if self.cfg.latent_reconloss:
                client_embedding = self.client_embeddings[client_idxs] + (self.client_embeddings[gt_client_idxs] - self.client_embeddings[client_idxs]).detach()
            else:
                client_embedding = self.client_embeddings[gt_client_idxs]

            

        z_combined = torch.cat((z, client_embedding), dim=-1)

        # Decoder
        recon_x = self.decoder(z_combined)
        result_dict = {
            "mu": mu,
            "log_var": log_var,
            "z": z,
            "recon_x": recon_x,
            "client_class_output": class_output,
            "client_idxs": client_idxs,
            "prob_class_output": prob_class_output
        }
        return result_dict

        # if return_classfier_output:
        #     return recon_x, mu, log_var, z, class_output
        # else:
        #     return recon_x, mu, log_var, z

    def encoder_forward(self, x):
        x = x.view(-1, 784)
        h = self.encoder(x)
        mu, log_var = h[:, :self.latent_dim], h[:, self.latent_dim:]
        z = self.reparameterize(mu, log_var)
        return mu, log_var, z



    def classifier_forward(self, c):
        class_output = self.cosine_classifier(c)
        return class_output

    def decoder_forward(self, z, client_idx):
        assert(client_idx is not None)
        if type(client_idx) == int:
            gt_client_idxs = torch.ones(z.size(0),dtype=torch.long).to(self.cfg.device) * client_idx
            client_embedding = self.client_embeddings[gt_client_idxs]
        else:
            # when client_idx is a tensor
            gt_client_idxs = client_idx
            client_embedding = self.client_embeddings[gt_client_idxs]
        z_combined = torch.cat((z, client_embedding), dim=-1)
        return self.decoder(z_combined)