import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import itertools

from sklearn.mixture import GaussianMixture


class CifarEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(CifarEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(8 * 8 * 16, latent_dim)
        self.fc_bn1 = nn.BatchNorm1d(latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.relu = nn.ReLU()

    def encode(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv4 = self.relu(self.bn4(self.conv4(conv3))).view(-1, 8 * 8 * 16)

        fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))
        return self.fc21(fc1)

    def forward(self, x):
        out = self.encode(x)
        return out


class Encoder(nn.Module):
    def __init__(self, input_dim, inter_dims, hid_dim):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(input_dim, 400),
                                     nn.ReLU(True),
                                     nn.Linear(400, 400),
                                     nn.ReLU(True),
                                     nn.Linear(400, 400),
                                     nn.ReLU(True))

        self.fc = nn.Linear(400, hid_dim)

    def forward(self, x):
        e = self.encoder(x)
        mu = self.fc(e)
        return mu


class Decoder(nn.Module):
    def __init__(self, input_dim, inter_dim, hid_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(hid_dim, inter_dim),
                                     nn.LeakyReLU(0.2, True),
                                     nn.Linear(inter_dim, input_dim),
                                     nn.Sigmoid())

    def forward(self, z):
        x_pro = self.decoder(z)
        return x_pro


def pre_train_gmm(x_dim, z_dim, gh_dim, dataloader, nclusters, nepoch, device, decoder=None, dataname='mnist'):
    recon_loss_fcn = nn.MSELoss()
    # cls_loss_fcn = nn.NLLLoss(reduction="mean")
    encoder = Encoder(x_dim, gh_dim, z_dim).to(device)
    if dataname.lower() == "cifar10":
        encoder = CifarEncoder(z_dim).to(device)
    if decoder is None:
        decoder = Decoder(x_dim, gh_dim, z_dim).to(device)

    optimizer = optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()),
                           lr=0.002, weight_decay=0.001)

    total_niter = int(dataloader.num_obs / 64) * nepoch
    for it in range(total_niter + 1):
        blobs = dataloader.forward()
        feat_data = torch.from_numpy(blobs['data']).float().to(device)  # image data

        z = encoder(feat_data)
        recon_feas = decoder(z,1)
        loss = recon_loss_fcn(recon_feas, feat_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_niter = int(dataloader.num_obs / 64)
    rep = []
    labels = []
    for it in range(total_niter+1):
        blobs = dataloader.forward()
        feat_data = torch.from_numpy(blobs['data']).float().to(device)  # image data
        labels.append(blobs['labels'])
        z = encoder(feat_data)
        rep.append(z.detach())

    rep = torch.cat(rep, dim=0)
    labels = np.concatenate(labels)
    gmm = GaussianMixture(n_components=nclusters, covariance_type='diag', reg_covar=1e-4)
    pre = gmm.fit_predict(rep.detach().cpu().numpy())
    acc = np.equal(pre, labels).sum() / pre.shape[0]
    print(f"Pre-train acc: {acc*100: .4f}")
    # print('Acc={:.4f}%'.format(cluster_acc(pre, Y)[0] * 100))
    # pi_ = torch.from_numpy(gmm.weights_).float()
    mu_c = torch.from_numpy(gmm.means_).float()
    log_sigma2_c = torch.log(torch.from_numpy(gmm.covariances_).float())

    return mu_c, log_sigma2_c


def gaussian_pdfs_log(x, mus, log_sigma2s, nclusters):
    G = []
    for c in range(nclusters):
        G.append(gaussian_pdf_log(x, mus[c], log_sigma2s[c]).view(-1, 1))
    return torch.cat(G, 1)


def gaussian_pdf_log(x, mu, log_sigma2):
    pi_constant = np.log(np.pi * 2)
    return -0.5 * (torch.sum(pi_constant + log_sigma2 + (x - mu).pow(2) / torch.exp(log_sigma2), dim=1))

