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

from gen_model import FeaturesGenerator
from classifier import eval_model
from dataset_GBU import StreamDataLayer, DATA_LOADER, FeatDataLayer


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
    opt.X_dim = dataset.feature_dim
    opt.Z_dim = opt.latent_dim
    num_k = 10

    # log_dir, cl_acc_dir = init_log()
    num_limit_component = 10
    energy_thres = 3.
    buffer_limit = 10
    cur_buffer_size = 0
    seen_label = []
    poor_data_buffer = []
    poor_data_labels = []
    poor_data_z = []
    data_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    result_metric = Result()
    ### Initialized trainable and inferable params here
    netG = FeaturesGenerator(opt.Z_dim, num_k, opt.X_dim).to(device)
    netG.apply(weights_init)
    print(netG)
    mus = torch.randn(num_limit_component, opt.Z_dim).float().to(device)
    logsigma = torch.zeros(num_limit_component, opt.Z_dim).float().to(device)

    data_loader = StreamDataLayer(dataset.train_label, dataset.train_feature, 64)
    optimizer_g = optim.Adam(netG.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    replay_stats = None
    while data_loader.has_next_batch():
        ### Get current stream of data
        cur_blob = data_loader.get_batch_data()
        cur_batch_feas = cur_blob["data"]
        cur_batch_labels = cur_blob["labels"]
        Z = torch.randn(cur_batch_feas.size(0), opt.Z_dim)
        Z = Z.float().to(device)
        ### Identify new component
        # batch_labels = torch.unique(cur_batch_labels)
        # for alab in batch_labels:
        #     if alab.item() not in seen_label:
        #         seen_label += [alab.item()]
        #         data_count[alab.item()] = 0
        #     else:
        #         init_new_comp = False

        ### Update count
        # for alab in seen_label:
        #     mask = cur_batch_labels == alab
        #     count = torch.sum(mask)
        #     data_count[alab] += count.item()

        ### TODO: Identify poorly-explained data
        poor_data_buffer.append(cur_batch_feas)
        poor_data_z.append(Z)
        poor_data_labels.append(cur_batch_labels)
        cur_buffer_size += 100

        ### Train model on data buffer
        if cur_buffer_size >= buffer_limit:
            print("============= Training poor data buffer. ============")
            train_x = torch.cat(poor_data_buffer, dim=0)
            train_y = torch.cat(poor_data_labels, dim=0)
            train_z = torch.cat(poor_data_z, dim=0)
            if replay_stats is not None:
                train_x = torch.cat([replay_stats[0], train_x], dim=0)
                train_y = torch.cat([replay_stats[1], train_y], dim=0)
                train_z = torch.cat([replay_stats[2], train_z], dim=0)
            print(f"train_x shape: {train_x.size()}")
            print(f"train_y shape: {train_y.size()}")
            print(f"Seen labels: {seen_label}")
            train_dataloader = FeatDataLayer(train_y.numpy(), train_x.numpy(), 64)
            total_iter = train_dataloader.num_obs // 64 * opt.nepoch
            ### EM_STEP
            for it in range(total_iter):
                mb_blob = train_dataloader.forward()
                x_mb = torch.from_numpy(mb_blob["data"]).to(device)
                idx_mb = mb_blob["idx"]
                y_mb = torch.from_numpy(mb_blob["labels"]).long()
                z_mb = train_z[idx_mb].to(device)
                z_mb.requires_grad_()

                optimizer_z = torch.optim.Adam([z_mb], lr=opt.lr, weight_decay=opt.weight_decay)
                scheduler_z = optim.lr_scheduler.StepLR(optimizer_z, step_size=5, gamma=0.97)
                batch_loss = 0
                for em_step in range(2):
                    optimizer_g.zero_grad()
                    one_hot_y = torch.eye(num_k)[y_mb]
                    recon_x = netG(z_mb, one_hot_y.to(device))
                    recon_loss = get_recon_loss(recon_x, x_mb, opt.sigma)  # Reconstruction Loss

                    # log_pdfs = get_prior_loss_mm(z_mb, mus, logsigma)
                    log_pdfs = get_prior_loss_mm(netG, z_mb, x_mb, num_k)
                    entropy_loss = get_entropy_loss(log_pdfs, one_hot_y.to(device))  # Entropy Loss

                    prior_loss = get_prior_loss(z_mb, mus[y_mb], logsigma[y_mb])

                    gloss = recon_loss + prior_loss + entropy_loss
                    gloss /= x_mb.size(0)
                    gloss.backward()
                    optimizer_g.step()
                    srmc_loss = 0

                    for _ in range(opt.langevin_step):
                        optimizer_z.zero_grad()
                        u_tau = torch.randn(z_mb.size(0), opt.Z_dim).float().to(device)

                        one_hot_y = torch.eye(num_k)[y_mb]
                        recon_x = netG(z_mb, one_hot_y.to(device))
                        recon_loss = get_recon_loss(recon_x, x_mb, opt.sigma)

                        # log_pdfs = get_prior_loss_mm(z_mb, mus, logsigma)
                        # entropy_loss = get_entropy_loss(log_pdfs, one_hot_y.to(device))

                        prior_loss = get_prior_loss(z_mb, mus[y_mb], logsigma[y_mb])

                        loss = recon_loss + prior_loss
                        loss /= x_mb.size(0)
                        loss = opt.langevin_s ** 2 / 2 * loss
                        loss.backward()
                        optimizer_z.step()
                        z_mb.data += u_tau * opt.langevin_s
                        srmc_loss += loss.detach()
                        scheduler_z.step()

                    train_z[idx_mb,] = z_mb.data
                    batch_loss += (srmc_loss / opt.langevin_step) + gloss.detach()
                batch_loss /= 2.

            ### Clear buffer after training
            cur_buffer_size = 0
            poor_data_z = []
            poor_data_labels = []
            poor_data_buffer = []

            ### Generate data from trained clusters
            print("Generate image")
            n_active = len(np.unique(train_dataloader.label))
            replay_stats = synthesize_features(netG, opt.nSample, n_active, num_k, opt.X_dim, opt.Z_dim)
            print("============= Done Training poor data buffer. ============\n")

        ### Evaluation
        if data_loader.get_cur_idx() % 640 == 0:
            netG.eval()
            eval_acc = eval_model(replay_stats[0], replay_stats[1],
                                  dataset.test_seen_feature, dataset.test_seen_label,
                                  opt.classifier_lr, device, 10)

            result_metric.update(1, eval_acc)
            print(f"Iter {data_loader.get_iteration()} Accuracy: {eval_acc: .2f}%")
            netG.train()


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


def get_prior_loss(z, mus, logsigma):
    log_pdf = 0.5 * torch.sum(np.log(2.0 * np.pi) + torch.pow(z, 2))
    return log_pdf


def get_prior_loss_mm(net, z, x, num_active):
    all_dist = []
    for ii in range(num_active):
        one_hot_y = torch.eye(10)[ii]
        one_hot_y = one_hot_y.repeat(x.size(0), 1).to(device)
        recon_x = net(z, one_hot_y)
        dist = -1.*torch.sum(torch.pow(recon_x - x, 2), dim=-1)
        all_dist.append(dist.unsqueeze(1))
    log_pdf = torch.cat(all_dist, dim=1)
    return log_pdf


def get_entropy_loss(logits, probs):
    log_q = torch.log_softmax(logits, dim=1)
    return torch.sum(-torch.sum(probs * log_q, dim=-1))


def save_model(it, netG, latent_state, replay_mem, random_seed, log, acc_log, mus, logsigma, fout):
    torch.save({
        'task_idx': it,
        'state_dict_G': netG.state_dict(),
        'train_z_state': latent_state.cpu().detach(),
        'replay_mem0': replay_mem[0].cpu().detach(),
        'replay_mem1': replay_mem[1].cpu().detach(),
        'replay_mem2': replay_mem[2].cpu().detach(),
        'random_seed': random_seed,
        'mus': mus.cpu().detach(),
        'logsigma': logsigma.cpu().detach(),
        'log_dir': log,
        'cl_acc_dir': acc_log
    }, fout)


def synthesize_features(netG, n_samples, n_active_comp, num_k, x_dim, latent_dim):
    gen_feat = torch.empty(n_active_comp*n_samples, x_dim).float()
    gen_label = np.zeros([0])
    replay_z = []
    with torch.no_grad():
        for ii in range(n_active_comp):
            one_hot_y = torch.eye(num_k)[ii]
            one_hot_y = one_hot_y.repeat(n_samples, 1)
            z = torch.randn(n_samples, latent_dim).to(device)
            G_sample = netG(z, one_hot_y.to(device))
            gen_feat[ii * n_samples:(ii + 1) * n_samples] = G_sample
            gen_label = np.hstack((gen_label, np.ones([n_samples]) * ii))
            replay_z.append(z)

    gen_label = torch.from_numpy(gen_label).long()
    replay_z = torch.cat(replay_z, dim=0)
    return gen_feat, gen_label, replay_z


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
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    for _ in range(opt.repeat):
        train()
