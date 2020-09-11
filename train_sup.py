import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.distributions as dist

import glob
import json
import argparse
import os
import random
import numpy as np
from time import gmtime, strftime
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

from classifier import eval_batch, eval_model
from dataset_GBU import FeatDataLayer, DATA_LOADER, StreamDataLayer
from pre_train import pre_train_gmm

parser = argparse.ArgumentParser()
parser.add_argument('--repeat', type=int, default=1, help='number of repeats for experiment')
parser.add_argument('--dataset', default='AWA1', help='dataset: CUB, AWA1, AWA2, SUN')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--image_embedding', default='res101', type=str)
parser.add_argument('--class_embedding', default='att', type=str)
parser.add_argument('--task_split_num', type=int, default=5, help='number of task split')

parser.add_argument('--nepoch', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train generator')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
parser.add_argument('--nSample', type=int, default=300, help='number features to generate per class')
parser.add_argument('--nSample_replay', type=int, default=100, help='number features for replay')

parser.add_argument('--resume', type=str, help='the model to resume')
parser.add_argument('--disp_interval', type=int, default=20)
parser.add_argument('--save_task', type=int, default=0)
parser.add_argument('--evl_interval', type=int, default=200)
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--latent_dim', type=int, default=10, help='dimension of latent z')
parser.add_argument('--gh_dim', type=int, default=2048, help='dimension of hidden layer in generator')
parser.add_argument('--latent_var', type=float, default=1, help='variance of prior distribution z')

parser.add_argument('--sigma', type=float, default=0.1, help='variance of random noise')
parser.add_argument('--sigma_U', type=float, default=1, help='variance of U_tau')
parser.add_argument('--langevin_s', type=float, default=0.1, help='s in langevin sampling')
parser.add_argument('--langevin_step', type=int, default=5, help='langevin step in each iteration')

parser.add_argument('--Knn', type=int, default=20, help='K value')
parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
opt = parser.parse_args()

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
device = f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu"

print('Running parameters:')
print(json.dumps(vars(opt), indent=4, separators=(',', ': ')))
print(device)


# our generator
class ConditionalGenerator(nn.Module):
    def __init__(self, opt, num_k):
        super(ConditionalGenerator, self).__init__()
        self.main = nn.Sequential(nn.Linear(opt.Z_dim + num_k, 400),
                                  nn.ReLU(True),
                                  nn.Linear(400, 400),
                                  nn.ReLU(True),
                                  nn.Linear(400, 400),
                                  nn.ReLU(True),
                                  nn.Linear(400, opt.X_dim),
                                  nn.Sigmoid())

    def forward(self, z, c):
        in_vec = torch.cat([z, c], dim=1)
        output = self.main(in_vec)
        return output


def init_log():
    out_dir = f'out/stream/{opt.dataset}/nreplay-{opt.nSample_replay}_sigma-{opt.sigma}_langevin_s-{opt.langevin_s}_' \
              f'step-{opt.langevin_step}_nepoch-{opt.nepoch}'
    os.makedirs(out_dir, exist_ok=True)
    print("The output dictionary is {}".format(out_dir))

    # log_dir = out_dir + '/log_{}.txt'.format(opt.dataset)
    it = 1
    while os.path.isfile(f"{out_dir}/log_it{it:02d}.txt"):
        it += 1
    log_dir = f"{out_dir}/log_it{it:02d}.txt"
    cl_acc_dir = f"{out_dir}/cl_acc_it{it:02d}.txt"
    with open(log_dir, 'w') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')
    with open(cl_acc_dir, "w") as f:
        f.write("")
    return log_dir, cl_acc_dir


def train():
    """
    Supervised sequential class learning without task boundary.
    - Able to use class label for training and component organization.
    - One component for one class.
    """
    dataset = DATA_LOADER(opt)
    opt.C_dim = dataset.att_dim
    opt.X_dim = dataset.feature_dim
    opt.Z_dim = opt.latent_dim
    opt.y_dim = dataset.ntrain_class

    log_dir, cl_acc_dir = init_log()
    num_limit_component = 10
    num_active_component = 10
    energy_thres = 3.
    buffer_limit = 100
    cur_buffer_size = 0
    init_new_comp = True
    seen_label = []
    poor_data_buffer = []
    poor_data_labels = []
    poor_data_pi = []
    poor_data_z = []
    data_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    result_metric = Result()
    ### Initialized trainable and inferable params here
    netG = ConditionalGenerator(opt, num_limit_component).to(device)
    netG.apply(weights_init)
    print(netG)
    mu_c = torch.randn(num_limit_component, opt.Z_dim).float().to(device)
    logsigma_c = torch.zeros(num_limit_component, opt.Z_dim).float().to(device)

    data_loader = StreamDataLayer(dataset.train_label, dataset.train_feature, 64)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    replay_stats = None
    while data_loader.has_next_batch():
        ### Get current stream of data
        cur_blob = data_loader.get_batch_data()
        cur_batch_feas = cur_blob["data"]
        cur_batch_labels = cur_blob["labels"]
        ### Identify new component
        batch_labels = torch.unique(cur_batch_labels)
        for alab in batch_labels:
            if alab.item() not in seen_label:
                seen_label += [alab.item()]
                init_new_comp = True
                # num_active_component += 1
                data_count[alab.item()] = 0
            else:
                init_new_comp = False

        ### Update count
        for alab in seen_label:
            mask = cur_batch_labels == alab
            count = torch.sum(mask)
            data_count[alab] += count.item()

        ### Identify poorly-explained data
        task_stats = get_class_stats_imp(mu_c, logsigma_c, np.arange(10))
        energy_level = eval_batch(netG, cur_batch_feas, task_stats, opt.Z_dim, 128, opt.sigma,
                                  10, num_limit_component, device)
        train_z = loc_init_z(task_stats, seen_label, opt.Z_dim, cur_batch_labels)
        if (data_loader.get_iteration() % 10) == 0:
            print(f"avg energy level: {energy_level.mean()}")
        mask = energy_level >= energy_thres
        poor_data_buffer.append(cur_batch_feas[mask])
        poor_data_z.append(train_z[mask])
        poor_data_labels.append(cur_batch_labels[mask])
        # poor_data_pi.append(prior_weights[mask])
        cur_buffer_size += torch.sum(mask)

        ### Train model on data buffer
        if cur_buffer_size >= buffer_limit:
            print("============= Training poor data buffer. ============")
            train_x = torch.cat(poor_data_buffer, dim=0).to(device)
            train_y = torch.cat(poor_data_labels, dim=0).to(device)
            Z = torch.cat(poor_data_z, dim=0).to(device)
            if replay_stats is not None:
                train_x = torch.cat([replay_stats[0], train_x], dim=0)
                train_y = torch.cat([replay_stats[1].to(device), train_y], dim=0)
                Z = torch.cat([replay_stats[2].to(device), Z], dim=0)
            prior_weights = torch.ones(train_x.size(0), 10) / len(seen_label)
            prior_weights = prior_weights.float().to(device)

            # task_mu = mu_c[:len(seen_label)]
            task_mu = mu_c
            task_mu.requires_grad_()
            # task_logsigma = logsigma_c[:len(seen_label)]
            task_logsigma = logsigma_c
            task_logsigma.requires_grad_()

            print(f"train_x shape: {train_x.size()}")
            print(f"train_y shape: {train_y.size()}")
            print(f"Seen labels: {seen_label}")
            train_dataloader = FeatDataLayer(train_y, train_x, 64)
            total_iter = train_dataloader.num_obs // 64 * opt.nepoch*len(seen_label)
            ### EM_STEP
            for it in range(total_iter):
                mb_blob = train_dataloader.forward()
                x_mb = mb_blob["data"]
                idx_mb = mb_blob["idx"]
                y_mb = mb_blob["labels"]
                z_mb = Z[idx_mb].to(device)
                z_mb.requires_grad_()
                pi_weights = prior_weights[idx_mb]
                optimizer_z = torch.optim.Adam([z_mb, task_mu, task_logsigma], lr=opt.lr, weight_decay=opt.weight_decay)
                for em_step in range(1):
                    # infer z
                    netG.train()
                    for _ in range(opt.langevin_step):
                        U_tau = torch.ones(z_mb.size()).normal_(0, opt.sigma_U).float().to(device)
                        pred = netG(z_mb, pi_weights)
                        recon_loss = get_recon_loss(pred, x_mb, opt.sigma)
                        entropy_loss, class_pred_prob = get_entropy_loss(z_mb, task_mu, task_logsigma, y_mb)
                        # entropy_loss, class_pred_prob = get_weights(z_mb, x_mb, pi_weights, y_mb,
                        #                                             netG, opt.sigma, num_active_component)
                        # prior_loss = get_prior_loss_mm(z_mb, task_mu, task_logsigma, class_pred_prob)
                        prior_loss = get_prior_loss(z_mb, task_mu, task_logsigma, class_pred_prob, True)
                        loss = recon_loss + prior_loss + entropy_loss
                        scaled_loss = (opt.langevin_s ** 2) * loss * 0.5
                        optimizer_z.zero_grad()
                        scaled_loss.backward()
                        # torch.nn.utils.clip_grad_norm_([Z], 5.)
                        # torch.nn.utils.clip_grad_norm_([task_mu, task_logsigma], 5.)
                        optimizer_z.step()
                        z_mb.data += opt.langevin_s * U_tau
                        _, class_pred_prob = get_entropy_loss(z_mb, task_mu, task_logsigma, y_mb)
                        # _, class_pred_prob = get_weights(z_mb, x_mb, pi_weights, y_mb,
                        #                                  netG, opt.sigma, len(seen_label))
                        pi_weights = class_pred_prob.detach()
                        optimizer_z.zero_grad()

                    # update w
                    for _ in range(1):
                        pred = netG(z_mb, pi_weights)
                        recon_loss = get_recon_loss(pred, x_mb, opt.sigma)
                        entropy_loss, class_pred_prob = get_entropy_loss(z_mb, task_mu, task_logsigma, y_mb)
                        # entropy_loss, class_pred_prob = get_weights(z_mb, x_mb, pi_weights, y_mb,
                        #                                             netG, opt.sigma, num_active_component)
                        # prior_loss = get_prior_loss_mm(z_mb, task_mu, task_logsigma, class_pred_prob)
                        prior_loss = get_prior_loss(z_mb, task_mu, task_logsigma, class_pred_prob, True)
                        loss = recon_loss + prior_loss + entropy_loss
                        # loss = recon_loss + prior_loss
                        optimizerG.zero_grad()
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(netG.parameters(), 5.)
                        optimizerG.step()
                        _, class_pred_prob = get_entropy_loss(z_mb, task_mu, task_logsigma, y_mb)
                        # _, class_pred_prob = get_weights(z_mb, x_mb, pi_weights, y_mb,
                        #                                  netG, opt.sigma, len(seen_label))
                        pi_weights = class_pred_prob.detach()
                        optimizerG.zero_grad()

                        if ((it+1) % 5) == 0:
                            # TODO: try evaluate on the train_x
                            class_pred = torch.argmax(class_pred_prob, dim=1)
                            corr = torch.sum(class_pred.eq(y_mb))
                            acc = 100.*corr / y_mb.size(0)
                            print(f"inner iter {it+1}/{total_iter}, loss: {loss:.4f}, acc: {acc:.4f}")

                Z[idx_mb] = z_mb.data
                prior_weights[idx_mb] = pi_weights.data
            ### Clear buffer after training
            cur_buffer_size = 0
            poor_data_z = []
            poor_data_labels = []
            poor_data_buffer = []
            # poor_data_pi = []
            # mu_c[:len(seen_label)] = task_mu.data
            # logsigma_c[:len(seen_label)] = task_logsigma.data
            mu_c = task_mu.data
            logsigma_c = task_logsigma.data
            ### Generate data from trained clusters
            print("Generate image")
            task_stats = get_class_stats(Z, train_y.cpu())
            if len(seen_label) > 1:
                prior_pi = torch.tensor([data_count[ii] for ii in range(len(seen_label)-1)]).float().to(device)
                init_new_comp = False
            else:
                prior_pi = torch.tensor([data_count[ii] for ii in range(len(seen_label))]).float().to(device)
            prior_pi = prior_pi / torch.sum(prior_pi)
            replay_stats = synthesize_features(netG, opt.Z_dim, prior_pi, 100, task_stats)
            print("============= Done Training poor data buffer. ============\n")

        ### Evaluation
        if data_loader.get_cur_idx() % 640 == 0:
            valid_data = []
            valid_labels = []
            for alab in seen_label:
                mask = dataset.train_label == alab
                valid_data.append(dataset.train_feature[mask])
                valid_labels.append(dataset.train_label[mask])
            valid_labels = torch.cat(valid_labels, dim=0)
            valid_data = torch.cat(valid_data, dim=0)
            prior_pi = torch.tensor([data_count[ii] for ii in range(len(seen_label))]).float().to(device)
            prior_pi = prior_pi / torch.sum(prior_pi)
            with torch.no_grad():
                # TODO: check if can use mu_c
                # task_stats = get_class_stats(Z, train_y.cpu())
                task_stats = get_class_stats_imp(mu_c, logsigma_c, np.arange(len(seen_label)))
                acc = eval_model(netG, valid_data, valid_labels, 128, prior_pi, len(seen_label), opt.Z_dim,
                                 opt.sigma, task_stats, device)
            result_metric.update(1, acc)
            print(f"Iter {data_loader.get_iteration()} Accuracy: {acc: .2f}%")
        netG.train()
        print_statement = False
    # result_metric.log_results(cl_acc_dir)


def log_print(s, log, print_str=True):
    if print_str:
        print(s)
    with open(log, 'a') as f:
        f.write(s + '\n')


def log_list(res_arr, log):
    num_item = len(res_arr)
    with open(log, "a") as f:
        for ii, res in enumerate(res_arr):
            if ii == num_item - 1:
                f.write(f"{res}\n")
            else:
                f.write(f"{res},")


def get_recon_loss(pred, x, sigma):
    recon_loss = 1/(2 * sigma ** 2) * torch.pow(x - pred, 2).sum()
    return recon_loss


def get_prior_loss(z, mus, log_sigma, class_prob, logits=False):
    # TODO: include weights calculation.
    pi_const = np.log(2.0 * np.pi)
    labels = class_prob
    if logits:
        labels = torch.argmax(class_prob, dim=1).detach()
    var_term = log_sigma[labels]
    dist_term = torch.pow(z - mus[labels], 2) / torch.exp(log_sigma[labels])
    dist_term = torch.sum(dist_term + var_term)
    loss = 0.5 * (pi_const + dist_term)
    return loss


def get_prior_loss_mm(z, mus, log_sigma, class_prob):
    """
    :param z: (B, z_dim)
    :param mus: (num_component, z_dim)
    :param log_sigma: (num_component, z_dim)
    :param class_prob: (B, num_component)
    :return:
    """
    pi_const = np.log(2.0 * np.pi)
    num_component = mus.size(0)
    z_expand = z.unsqueeze(1).repeat(1, num_component, 1)
    mu_expand = mus.unsqueeze(0).repeat(z.size(0), 1, 1)
    sd_expand = torch.exp(log_sigma.unsqueeze(0).repeat(z.size(0), 1, 1))
    dist_term = torch.sum(torch.pow(z_expand - mu_expand, 2) / sd_expand, dim=2)  # size (B, nc)
    loss = (0.5 * (pi_const + dist_term))
    # loss = torch.pow(z, 2) * 0.5
    return loss.sum()


def get_entropy_loss(z, mus, log_sigma, labels):
    loss_fcn = nn.CrossEntropyLoss(reduction='sum')
    num_mu = mus.size(0)
    latent_dim = mus.size(1)
    diag_mat = torch.eye(latent_dim).unsqueeze(0).to(device) * torch.exp(log_sigma).unsqueeze(1)
    log_pdf = dist.MultivariateNormal(mus, diag_mat).log_prob(z.unsqueeze(1).repeat(1, num_mu, 1))
    yita_c = log_pdf
    loss = loss_fcn(yita_c, labels)
    return loss, torch.softmax(yita_c, dim=1)


def get_weights(z, x, weights, labels, net, sigma, num_component):
    loss_fcn = nn.CrossEntropyLoss(reduction='sum')
    log_prob = []
    for ii in range(10):
        one_hot_y = torch.eye(10)
        one_hot_y = one_hot_y[ii].unsqueeze(0).repeat(z.size(0), 1)
        pred = net(z, one_hot_y.to(device))
        recon_loss = -1/(2*sigma**2) * torch.pow(x - pred, 2).sum(dim=1)
        log_prob.append(recon_loss.unsqueeze(1))
    log_prob = torch.cat(log_prob, dim=1)
    log_weights = torch.softmax(log_prob, dim=1)
    loss = loss_fcn(log_prob, labels)
    return loss, log_weights


def save_model(it, netG, replay_stats, random_seed, log, acc_log, mus, logsigma, fout):
    torch.save({
        'it': it + 1,
        'state_dict_G': netG.state_dict(),
        'replay_mem0': replay_stats[0].cpu().detach(),
        'replay_mem1': replay_stats[1].cpu().detach(),
        'replay_mem2': replay_stats[2].cpu().detach(),
        'random_seed': random_seed,
        'mus': mus.cpu().detach(),
        'logsigma': logsigma.cpu().detach(),
        'log_dir': log,
        'cl_acc_dir': acc_log
    }, fout)


def get_class_stats(latent, labels, unique_label=None):
    assert latent.size(0) == labels.shape[0]
    if unique_label is None:
        unique_label = torch.unique(labels)
    train_stats = {}
    # train_loc = torch.zeros(len(unique_label), latent.size(1))
    for ii, label in enumerate(unique_label):
        mask = labels == label
        z_samp = latent[mask]
        var, loc = torch.var_mean(z_samp, dim=0)
        train_stats[ii] = (loc, var)
    return train_stats


def get_class_stats_imp(locs, sigmas, unique_label):
    # unique_label = torch.unique(labels)
    train_stats = {}
    for ii, label in enumerate(unique_label):
        train_stats[ii] = (locs[label].detach(), sigmas[label].detach())
    return train_stats


def loc_init_z(locs, unique_labels, latent_dim, data_input):
    train_z = torch.randn(data_input.size(0), latent_dim)
    for ii, task_label in enumerate(unique_labels):
        mask = data_input == task_label
        diag_mat = torch.exp(locs[task_label][1].to("cpu")) * torch.eye(latent_dim)
        train_z[mask] = dist.MultivariateNormal(locs[task_label][0].to("cpu"),
                                                diag_mat.to("cpu")).sample([torch.sum(mask).item()])
    return train_z


def loc_init_mu(prev_mu, cur_mu):
    ata = torch.matmul(prev_mu, prev_mu.t())
    inv_ata = torch.inverse(ata)
    proj_mat = torch.matmul(prev_mu.t(), torch.matmul(inv_ata, prev_mu))
    new_mu = cur_mu - torch.matmul(proj_mat, cur_mu.t()).t()
    return new_mu


def synthesize_features(netG, z_dim, pi_weights, num_samples, class_stats):
    """
    Generate samples for replay
    :param netG: generator network
    :param z_dim: latent dimensions
    :param pi_weights: weights of active components
    :param num_samples: number of samples to generate
    :param class_stats: dictionary of component statistics
    :return:
    """
    gen_feas = []
    gen_labels = []
    gen_z = []
    num_component = pi_weights.size(0)
    with torch.no_grad():
        for ii in range(num_component):
            # nsamp = int(num_samples * pi_weights[ii])
            nsamp = num_samples
            # z = torch.randn(nsamp, z_dim)
            z = dist.Normal(class_stats[ii][0], torch.ones(z_dim).to(device)).sample([nsamp])
            one_hot_y = torch.eye(10)
            one_hot_y = one_hot_y[ii].unsqueeze(0).repeat(nsamp, 1)
            gen_sample = netG(z.to(device), one_hot_y.to(device))
            gen_feas.append(gen_sample)
            gen_labels.append(torch.ones(nsamp) * ii)
            gen_z.append(z)
    gen_feas = torch.cat(gen_feas, dim=0)
    gen_labels = torch.cat(gen_labels, dim=0)
    gen_z = torch.cat(gen_z, dim=0)
    return gen_feas, gen_labels.long(), gen_z


def samp_features(trained_labels, feas, latent_rep, n_samples):
    unique_trained_labels = np.unique(trained_labels)
    nclass = len(unique_trained_labels)
    gen_feat = []
    gen_label = []
    gen_z = []
    for ii in range(nclass):
        label = unique_trained_labels[ii]
        mask = trained_labels == label
        subset_feas = feas[mask]
        subset_z = latent_rep[mask]
        subset_labels = trained_labels[mask]
        if subset_feas.shape[0] < n_samples:
            subsamp_idx = np.random.choice(np.arange(np.sum(mask)), subset_feas.shape[0], replace=False)
        else:
            subsamp_idx = np.random.choice(np.arange(np.sum(mask)), n_samples, replace=False)
        gen_feat.append(torch.from_numpy(subset_feas[subsamp_idx]))
        gen_label.append(subset_labels[subsamp_idx])
        gen_z.append(subset_z[subsamp_idx])
    gen_feat = torch.cat(gen_feat, dim=0)
    gen_label = np.concatenate(gen_label)
    gen_z = torch.cat(gen_z, dim=0)
    return gen_feat, torch.from_numpy(gen_label.astype(int)), gen_z.detach()


def eval_knn(gen_feat, gen_label, test_feats, test_labels, knn):
    # cosince predict K-nearest Neighbor
    n_test_sample = test_feats.shape[0]
    index_boundary = [ii for ii in range(0, n_test_sample, 64)]
    if index_boundary[-1] < (n_test_sample):
        index_boundary += [n_test_sample]

    pred_list = []
    for ii in range(1, len(index_boundary)):
        start_idx = index_boundary[ii - 1]
        end_idx = index_boundary[ii]
        num_samp = test_feats[start_idx:end_idx].shape[0]
        sim = cosine_similarity(test_feats[start_idx:end_idx], gen_feat)
        # only count first K nearest neighbor
        idx_mat = np.argsort(-1 * sim, axis=1)[:, 0:knn]
        label_mat = gen_label[idx_mat.flatten()].reshape((num_samp, -1))
        preds = np.zeros(num_samp)
        for i in range(num_samp):
            label_count = Counter(label_mat[i]).most_common(1)
            preds[i] = label_count[0][0]
        pred_list.append(preds)
    pred_list = np.concatenate(pred_list)
    acc = eval_MCA(pred_list, test_labels) * 100
    return acc


def eval_MCA(preds, y):
    cls_label = np.unique(y)
    acc = list()
    for i in cls_label:
        acc.append((preds[y == i] == i).mean())
    return np.asarray(acc).mean()


class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_iter = 0.0
        self.acc_list = []
        self.iter_list = []
        self.save_model = False

        self.task_acc = []
        self.gzsl_acc = []

    def update(self, it, acc):
        self.acc_list += [acc]
        self.iter_list += [it]
        self.save_model = False
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_iter = it
            # self.save_model = True

    def update_task_acc(self, acc: list):
        self.task_acc.append(acc)

    def update_gzsl_acc(self, acc):
        self.gzsl_acc += [acc]

    def log_results(self, log):
        num_item = len(self.task_acc[-1])
        for each_res in self.task_acc:
            if len(each_res) < num_item:
                res_arr = each_res + [0.0] * (num_item - len(each_res))
            else:
                res_arr = each_res
            log_list(res_arr, log)


def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.normal_(m.weight.data, mean=0, std=0.02)
        init.constant_(m.bias, 0.0)


if __name__ == "__main__":
    for _ in range(opt.repeat):
        train()
