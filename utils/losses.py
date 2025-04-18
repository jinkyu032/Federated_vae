import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import gc # Garbage collector for explicit memory management
from sklearn.feature_selection import mutual_info_classif
__all__ = ['vae_loss', 'compute_loss']

# VAE loss function
def vae_loss(recon_x, x, mu = None, log_var = None, mu_target=0, reduction='mean', reconloss_only=False):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = None
    if not reconloss_only:
        KLD = -0.5 * torch.sum(2*mu_target*mu + 1 + log_var - mu.pow(2) - log_var.exp() - mu_target*mu_target)

    if reduction == 'mean':
        BCE /= x.size(0)
        if not reconloss_only:
            KLD /= x.size(0)

    return BCE, KLD

def celoss(x,y, reduction='mean', temp=1):
    criterion = nn.CrossEntropyLoss(reduction=reduction)
    loss = criterion(x/temp, y)
    return loss

# Compute loss for a data loader
def compute_loss(cfg, model, data_loader, device, mu_target=0, reduction='sum', compare_model = None, *args, **kwargs):
    model_list = []
    model.eval()
    model.to(device)
    total_loss_sum = 0
    recon_loss_sum = 0
    kl_loss_sum = 0
    correct = 0
    codebook_loss_sum = 0
    commitment_loss_sum = 0
    client_class_celoss = 0
    model_list.append(model)
    mean_var = 0
    if compare_model is not None:
        compare_model.eval()
        compare_model.to(device)
        mean_l2_compare = 0
        var_diff_compare = 0
        model_list.append(compare_model)
        mean_var_compare = 0
    

    def forward_modellist(model_list, *args, **kwargs):
        result_list = []
        for model in model_list:
            result = model(*args, **kwargs)
            result_list.append(result)
        return result_list



    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            if cfg.vq:
                #recon_batch, codebook_loss, commitment_loss = model(data, target)
                result = model(data, target)
                recon_batch, codebook_loss, commitment_loss = result['recon_x'], result['codebook_loss'], result['commitment_loss']
                recon_loss, _ = vae_loss(recon_batch, data, reconloss_only=True, reduction=reduction)
                if reduction == 'mean':
                    codebook_loss = codebook_loss.mean()
                    commitment_loss = commitment_loss.mean()
                elif reduction == 'sum':
                    codebook_loss = codebook_loss.sum()
                    commitment_loss = commitment_loss.sum()
                total_loss_sum += (recon_loss.item() + codebook_loss.item() + cfg.commitment_weight * commitment_loss.item())
                recon_loss_sum += recon_loss.item()
                codebook_loss_sum += codebook_loss.item()
                commitment_loss_sum += commitment_loss.item()
            else:
                if cfg.client_classifier:
                    if kwargs.get("oracle", False):
                        results = forward_modellist(model_list, data, kwargs["client_idx"], return_classfier_output=True) #model(data, kwargs["client_idx"], return_classfier_output=True)
                    else:
                        results = forward_modellist(model_list, data, return_classfier_output=True)#model(data, return_classfier_output=True)

                    result = results[0]
                    #recon_batch, mu, log_var, z, class_output = result['recon_x'], result['mu'], result['log_var'], result['z'], result['client_class_output']
                    recon_batch, mu, log_var, z, prob_class_output, client_idxs = result['recon_x'], result['mu'], result['log_var'], result['z'], result['prob_class_output'],  result['client_idxs']

                    client_target = torch.ones(data.size(0),dtype=torch.long).to(cfg.device) * kwargs["client_idx"]
                    embedding_loss = celoss(prob_class_output, client_target, reduction=cfg.reduction)
                    client_class_celoss += embedding_loss.sum().item()
                    predicted = client_idxs
                    correct += (predicted == kwargs["client_idx"]).sum().item()
                    total_loss_sum += embedding_loss.sum().item()


                elif cfg.use_classifier:
                    #recon_batch, mu, log_var, z, class_output = model(data, return_classfier_output=True)
                    #result = model(data, return_classfier_output=True)
                    results = forward_modellist(model_list, data, return_classfier_output=True)
                    result = results[0]
                    recon_batch, mu, log_var, z, class_output = result['recon_x'], result['mu'], result['log_var'], result['z'], result['client_class_output']
                    _, predicted = torch.max(class_output.data, 1)
                    correct += (predicted == target).sum().item()
                else:
                    #recon_batch, mu, log_var, z = model(data, target)
                    #result = model(data, target)
                    results = forward_modellist(model_list, data, target)
                    result = results[0]
                    recon_batch, mu, log_var, z = result['recon_x'], result['mu'], result['log_var'], result['z']
                recon_loss, kl_loss = vae_loss(recon_batch, data, mu, log_var, mu_target=mu_target, reduction = 'sum')
                total_loss_sum += (recon_loss.item() + cfg.kl_weight * kl_loss.item())
                recon_loss_sum += recon_loss.item()
                kl_loss_sum += kl_loss.item()
                mean_var += torch.mean(torch.exp(log_var), dim=1).sum().item() #bug torch.sum(torch.exp(log_var), dim=1).sum().item() until 04/17 23:59

            if compare_model is not None:
                compare_result = results[1]
                compare_recon_batch, compare_mu, compare_log_var, compare_z = compare_result['recon_x'], compare_result['mu'], compare_result['log_var'], compare_result['z']
                mean_l2_compare += torch.mean((mu - compare_mu) ** 2, dim=1).sum().item()
                var_diff_compare += torch.mean(torch.exp(log_var - compare_log_var), dim=1).sum().item()
                mean_var_compare += torch.mean(torch.exp(compare_log_var), dim=1).sum().item()
    result = {}
    result['total_loss'] = total_loss_sum / len(data_loader.dataset)
    result['recon_loss'] = recon_loss_sum / len(data_loader.dataset)
    result['mean_var'] = mean_var / len(data_loader.dataset)
    if cfg.vq:
        result['codebook_loss'] = codebook_loss_sum / len(data_loader.dataset)
        result['commitment_loss'] = commitment_loss_sum / len(data_loader.dataset)
    else:
        result['kl_loss'] = kl_loss_sum / len(data_loader.dataset)

    if cfg.client_classifier:
        result['distance_from_client_embedding'] = client_class_celoss / len(data_loader.dataset)
        result['client_classification_accuracy'] = 100 * correct / len(data_loader.dataset)

    if cfg.use_classifier:
        result['accuracy'] = 100 * correct / len(data_loader.dataset)
    
    if compare_model is not None:
        result['mean_l2_compare'] = mean_l2_compare / len(data_loader.dataset)
        result['var_diff_compare'] = var_diff_compare / len(data_loader.dataset)
        result['mean_var_compare'] = mean_var_compare / len(data_loader.dataset)
    return result



