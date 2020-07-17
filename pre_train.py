import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.mixture import GaussianMixture


def pre_train_gmm(net, train_z, train_att, dataloader, nclusters, nepoch, device):
    loss_fcn = nn.MSELoss()
    # entropy_loss = nn.CrossEntropyLoss()
    net.train()
    total_niter = int(dataloader.num_obs / 64) * nepoch
    for it in range(total_niter + 1):
        blobs = dataloader.forward()
        feat_data = torch.from_numpy(blobs['data']).float().to(device)  # image data
        labels = blobs['labels'].astype(int)  # class labels
        idx = blobs['idx'].astype(int)
        C = torch.tensor([train_att[i, :] for i in labels]).float().to(device)
        Z = train_z[idx].to(device)
        Z.requires_grad_()
        optimizer = optim.Adam(net.parameters(), lr=0.002, weight_decay=0.001)
        optimizer.add_param_group({"params": [Z]})

        pred = net(Z, C)
        loss = loss_fcn(pred, feat_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_z[idx] = Z.data

    gmm = GaussianMixture(n_components=nclusters, covariance_type='diag')

    pre = gmm.fit_predict(train_z.detach().cpu())
    # print('Acc={:.4f}%'.format(cluster_acc(pre, Y)[0] * 100))

    # pi_ = torch.from_numpy(gmm.weights_).float()
    mu_c = torch.from_numpy(gmm.means_).float()
    log_sigma2_c = torch.log(torch.from_numpy(gmm.covariances_).float())

    return mu_c, log_sigma2_c


def gaussian_pdfs_log(x, mus, log_sigma2s, nclusters):
    G=[]
    for c in range(nclusters):
        G.append(gaussian_pdf_log(x, mus[c], log_sigma2s[c]).view(-1, 1))
    return torch.cat(G, 1)


def gaussian_pdf_log(x, mu, log_sigma2):
    pi_constant = np.log(np.pi*2)
    return -0.5*(torch.sum(pi_constant+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2), dim=1))
