import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
import sys
import pdb


# p(y| x, z)
def infer_class(x, z, net, mus, logsigmas, sigma, device):
    """
    Assuming the model p(x,y,z) = p(x|z_y)p(z|y)p(y)
    """
    assert x.size(0) == z.size(0)
    latent_dim = mus.size(1)
    num_mu = mus.size(0)
    diag_mat = torch.eye(latent_dim).unsqueeze(0).to(device) * torch.exp(logsigmas).unsqueeze(1)
    gaussian_log_pdf = dist.MultivariateNormal(mus, diag_mat).log_prob(z.unsqueeze(1).repeat(1, num_mu, 1))
    # pred = net(z, 1)
    # obs_log_pdf = -1/(2*sigma**2) * torch.pow(x - pred, 2)  # (B, feature_size)
    log_pdf = gaussian_log_pdf
    class_prob = torch.softmax(log_pdf, dim=1)  # (B, num_mu)
    return class_prob


def eval_latent(net, feats, labels, latent_rep, mus, logsigmas, sigma, device):
    mb_size = 64
    n_test_sample = feats.shape[0]
    index_boundary = [ii for ii in range(0, n_test_sample, mb_size)]
    if index_boundary[-1] < n_test_sample:
        index_boundary += [n_test_sample]

    pred_prob = []
    with torch.no_grad():
        for ii in range(1, len(index_boundary)):
            start_idx = index_boundary[ii-1]
            end_idx = index_boundary[ii]
            x_mb = torch.from_numpy(feats[start_idx:end_idx]).to(device)
            z_mb = latent_rep[start_idx:end_idx]
            class_pred = infer_class(x_mb, z_mb, net, mus, logsigmas, sigma, device)
            pred_prob.append(class_pred)

    pred_class = torch.argmax(torch.cat(pred_prob, dim=0), dim=1)
    cor_mask = np.equal(labels, pred_class.cpu().numpy())
    num_correct = np.sum(cor_mask)
    acc = num_correct / cor_mask.shape[0]
    return acc * 100


def gen_latent(nsamples, mu, logsigma, latent_dim, device):
    diag_mat = torch.eye(latent_dim).to(device) * logsigma.unsqueeze(1)
    z = dist.MultivariateNormal(mu, diag_mat).sample([nsamples])
    return z


def eval_model_cls(net, feats, labels, nsamples, task_stats, latent_dim, sigma, device):
    mb_size = 64
    n_test_sample = feats.shape[0]
    index_boundary = [ii for ii in range(0, n_test_sample, mb_size)]
    if index_boundary[-1] < n_test_sample:
        index_boundary += [n_test_sample]

    # num_mu = mus.size(0)
    num_mu = len(task_stats.keys())
    data_log_pdfs = []
    for ii in range(1, len(index_boundary)):
        start_idx = index_boundary[ii-1]
        end_idx = index_boundary[ii]
        x_new = feats[start_idx:end_idx]  # (mb, in_dim)
        log_pdfs = []
        for ii in range(num_mu):
            z_samp = gen_latent(nsamples, task_stats[ii][0], task_stats[ii][1], latent_dim, device)  # (nsamples, latent_dim)
            x_pred = net(z_samp, 1)  # (nsamples, in_dim)
            x_new_expand = x_new.unsqueeze(1).repeat(1, nsamples, 1).to(device)
            x_pred_expand = x_pred.unsqueeze(0).repeat(x_new.size(0), 1, 1)
            obs_log_pdf = -1/(2*sigma**2) * torch.pow(x_new_expand - x_pred_expand, 2).mean(dim=1).mean(dim=1)  # (B, 1)
            log_pdfs.append(obs_log_pdf.detach().unsqueeze(1))
        log_pdfs = torch.cat(log_pdfs, dim=1)
        data_log_pdfs.append(log_pdfs)
    data_log_pdfs = torch.cat(data_log_pdfs, dim=0)
    class_pred = torch.argmax(torch.softmax(data_log_pdfs, dim=1), dim=1)
    cor_mask = np.equal(class_pred.cpu().numpy(), labels.numpy())
    acc = cor_mask.sum() / n_test_sample
    return acc*100
