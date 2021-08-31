import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
import sys
import pdb
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import svm
from collections import Counter

from models.gen_model import LinearCLS


def eval_knn(gen_feat, gen_label, test_feas, test_labels, Knn):
    # cosince predict K-nearest Neighbor
    n_test_sample = test_feas.shape[0]
    sim = cosine_similarity(test_feas, gen_feat)
    # only count first K nearest neighbor
    idx_mat = np.argsort(-1 * sim, axis=1)[:, 0:Knn]
    label_mat = gen_label[idx_mat.flatten()].reshape((n_test_sample,-1))
    preds = np.zeros(n_test_sample)
    for i in range(n_test_sample):
        label_count = Counter(label_mat[i]).most_common(1)
        preds[i] = label_count[0][0]
    acc = eval_MCA(preds, test_labels) * 100
    return acc


def eval_MCA(preds, y):
    cls_label = np.unique(y)
    acc = list()
    for i in cls_label:
        acc.append((preds[y == i] == i).mean())
    return np.asarray(acc).mean()


def train_classifier(feats, labels, lr_rate, device, num_classes=10):
    mb_size = 512
    linear_cls = LinearCLS(feats.size(1), num_classes).to(device)
    optimizer_cls = torch.optim.Adam(linear_cls.parameters(), lr=lr_rate, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_cls, T_max=40, eta_min=0.0005)
    cls_criterion = nn.NLLLoss()

    # num_iter = feats.size(0) // mb_size
    bd_indices = [ii for ii in range(0, feats.size(0), mb_size)]

    # Train Linear Classifier
    for epc in range(40):
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
        scheduler.step()
    return linear_cls


def eval_model(linear_cls, valid_feas, valid_label, device):
    mb_size=512
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
