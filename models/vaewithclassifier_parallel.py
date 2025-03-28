import torch
import torch.nn as nn
from utils.data import idx2onehot

__all__ = ['VAEWithClassifier']

class VAEWithClassifier_parallel(nn.Module):
    def __init__(self, hidden_dims=[512, 256], latent_dim=2, num_classes=20, batch_norm=False, cfg = None):
        super(VAEWithClassifier_parallel, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.batch_norm = batch_norm
        self.cfg = cfg

        # Unconditional Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]) if self.batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]) if self.batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], latent_dim * 2 + num_classes)  # 2 for mu, 2 for log_var
        )

        # # Classifier (takes latent z as input)
        # self.classifier = nn.Sequential(
        #     nn.Linear(latent_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, num_classes)
        # )

        # Decoder (takes concatenated latent and classifier output)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]) if self.batch_norm else nn.Identity(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]) if self.batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], 784),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, target = None,return_classfier_output=False):
        x = x.view(-1, 784)

        # Encoder
        h = self.encoder(x)
        mu, log_var = h[:, :self.latent_dim], h[:, self.latent_dim:2* self.latent_dim]
        z = self.reparameterize(mu, log_var)

        # Classifier
        class_output = h[:,2* self.latent_dim:]

        # Concatenate latent and classifier output
        #c = class_output  # or potentially idx2onehot(torch.argmax(class_output, dim=1), n=self.num_classes)
        if self.cfg.problabelfeatures:
            class_output = nn.functional.log_softmax(class_output, dim=1)
        c = class_output
        z_combined = torch.cat((z, c), dim=-1)

        # Decoder
        recon_x = self.decoder(z_combined)

        if return_classfier_output:
            return recon_x, mu, log_var, z, class_output
        else:
            return recon_x, mu, log_var, z

    def encoder_forward(self, x):
        x = x.view(-1, 784)
        h = self.encoder(x)
        mu, log_var = h[:, :self.latent_dim], h[:, self.latent_dim:]
        z = self.reparameterize(mu, log_var)
        return mu, log_var, z

    # def classifier_forward(self, z):
    #     class_output = self.classifier(z)
    #     return class_output

    def decoder_forward(self, z):
        #c = self.classifier_forward(z)
        z_combined = z#torch.cat((z, c), dim=-1)
        return self.decoder(z_combined)