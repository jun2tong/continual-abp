import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
import sys
import pdb
from scipy.optimize import linear_sum_assignment
from gen_model import LinearCLS


def cluster_acc(Y_pred, Y):
    # from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size(0) == Y.size(0)
    D = max(Y_pred.max(), Y.max()) + 1
    weights = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size(0)):
        weights[Y_pred[i], Y[i]] += 1
    ind = linear_sum_assignment(weights.max() - weights)
    # return sum([weights[i, j] for i, j in ind]) * 1.0 / Y_pred.size(0)
    return sum(weights[ind[0], ind[1]]) / Y_pred.size(0) * 100.


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
    diag_mat = torch.eye(latent_dim).to(device) * torch.ones_like(logsigma).unsqueeze(1).to(device)
    z = dist.MultivariateNormal(mu, diag_mat).sample([nsamples])
    return z


def eval_model_cls(net, feats, labels, nsamples, task_stats, latent_dim, sigma, device, dataname='mnist'):
    mb_size = 64
    n_test_sample = feats.shape[0]
    index_boundary = [ii for ii in range(0, n_test_sample, mb_size)]
    if index_boundary[-1] < n_test_sample:
        index_boundary += [n_test_sample]

    # num_mu = mus.size(0)
    num_mu = len(task_stats.keys())
    data_log_pdfs = []
    for idx in range(1, len(index_boundary)):
        start_idx = index_boundary[idx-1]
        end_idx = index_boundary[idx]
        x_new = feats[start_idx:end_idx]  # (mb, in_dim)
        log_pdfs = []
        for ii in range(num_mu):
            z_samp = gen_latent(nsamples, task_stats[ii][0], task_stats[ii][1], latent_dim, device)  # (nsamples, latent_dim)
            x_pred = net(z_samp, 1)  # (nsamples, in_dim)
            if dataname.lower() == 'cifar10':
                x_new_expand = x_new.unsqueeze(1).repeat(1, nsamples, 1, 1, 1).to(device)
                x_pred_expand = x_pred.unsqueeze(0).repeat(x_new.size(0), 1, 1, 1, 1)
                energy = torch.pow(x_new_expand - x_pred_expand, 2).mean(dim=1).view(x_new.size(0), -1).sum(dim=-1)
                obs_log_pdf = -1/(2*sigma**2) * energy
            else:
                x_new_expand = x_new.unsqueeze(1).repeat(1, nsamples, 1).to(device)
                x_pred_expand = x_pred.unsqueeze(0).repeat(x_new.size(0), 1, 1)
                obs_log_pdf = -1/(2*sigma**2) * torch.pow(x_new_expand - x_pred_expand, 2).mean(dim=1).sum(dim=1)  # (B, 1)
            log_pdfs.append(obs_log_pdf.detach().unsqueeze(1))
        log_pdfs = torch.cat(log_pdfs, dim=1)
        data_log_pdfs.append(log_pdfs)
    data_log_pdfs = torch.cat(data_log_pdfs, dim=0)
    class_pred = torch.argmax(torch.softmax(data_log_pdfs, dim=1), dim=1)
    cor_mask = np.equal(class_pred.cpu().numpy(), labels.numpy())
    acc = cor_mask.sum() / n_test_sample
    return acc*100


def eval_model(feats, labels, valid_feas, valid_label, lr_rate, device, num_classes=10):
    mb_size = 64
    linear_cls = LinearCLS(feats.size(1), num_classes).to(device)
    optimizer_cls = torch.optim.Adam(linear_cls.parameters(), lr=lr_rate, betas=(0.5, 0.999))
    cls_criterion = nn.NLLLoss()

    # num_iter = feats.size(0) // mb_size
    bd_indices = [ii for ii in range(0, feats.size(0), mb_size)]

    # Train Linear Classifier
    for epc in range(20):
        batch_indices = np.random.permutation(np.arange(feats.size(0)))
        for idx, iter_count in enumerate(bd_indices):
            batch_idx = batch_indices[iter_count:iter_count + mb_size]
            x_mb = feats[batch_idx].float().to(device)
            y_mb = labels[batch_idx].to(device)
            output = linear_cls(x_mb)
            loss = cls_criterion(output, y_mb)
            optimizer_cls.zero_grad()
            loss.backward()
            optimizer_cls.step()

    # num_iter = valid_feas.size(0) // mb_size
    bd_indices = [ii for ii in range(0, valid_feas.size(0), mb_size)]
    batch_indices = np.arange(valid_feas.size(0))
    corr = 0
    with torch.no_grad():
        for idx, iter_count in enumerate(bd_indices):
            batch_idx = batch_indices[iter_count:iter_count + mb_size]
            x_mb = valid_feas[batch_idx].float().to(device)
            y_mb = valid_label[batch_idx].to(device)
            output = linear_cls(x_mb)
            pred = torch.argmax(output, dim=1)
            corr += torch.sum(torch.eq(y_mb, pred))

    acc = 100 * corr.cpu().item() / float(valid_feas.size(0))

    return acc


def eval_batch(net, batch_feas, class_stats, latent_dim, nsamples, sigma,
               num_active_component, num_limit_comp, device):
    """
    Evaluate how to process this batch of data.
    :return:
    """
    energies = []
    with torch.no_grad():
        for ii in range(num_active_component):
            z_samp = dist.Normal(class_stats[ii][0],
                                 torch.ones(latent_dim).to(device)).sample([nsamples])
            one_hot_y = torch.eye(num_limit_comp)
            one_hot_y = one_hot_y[ii].unsqueeze(0).repeat(nsamples, 1)
            x_pred = net(z_samp.to(device), one_hot_y.to(device))  # (nsample, in_dim)
            x_new_expand = batch_feas.unsqueeze(1).repeat(1, nsamples, 1).to(device)
            x_pred_expand = x_pred.unsqueeze(0).repeat(batch_feas.size(0), 1, 1)
            obs_log_pdf = 1/(2 * sigma ** 2) * torch.pow(x_new_expand - x_pred_expand, 2).mean(dim=1).mean(dim=1)
            energies.append(obs_log_pdf.cpu().unsqueeze(1))

    energies = torch.cat(energies, dim=1)  # (B, num_component)
    return torch.sum(energies, dim=1)
