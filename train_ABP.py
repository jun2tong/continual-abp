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

from classifier import eval_latent, eval_model_cls
from dataset_GBU import FeatDataLayer, DATA_LOADER
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
parser.add_argument('--lr', type=float, default=0.002, help='learning rate to train generator')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
parser.add_argument('--nSample', type=int, default=300, help='number features to generate per class')
parser.add_argument('--nSample_replay', type=int, default=100, help='number features for replay')

parser.add_argument('--resume',  type=str, help='the model to resume')
parser.add_argument('--disp_interval', type=int, default=20)
parser.add_argument('--save_task', type=int, default=0)
parser.add_argument('--evl_interval',  type=int, default=200)
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--latent_dim', type=int, default=32, help='dimension of latent z')
parser.add_argument('--gh_dim',     type=int, default=2048, help='dimension of hidden layer in generator')
parser.add_argument('--latent_var', type=float, default=1, help='variance of prior distribution z')

parser.add_argument('--sigma',   type=float, default=0.1, help='variance of random noise')
parser.add_argument('--sigma_U', type=float, default=1,   help='variance of U_tau')
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
    def __init__(self, opt):
        super(ConditionalGenerator, self).__init__()
        # self.main = nn.Sequential(nn.Linear(opt.Z_dim, opt.gh_dim),
        #                           nn.LeakyReLU(0.2, True),
        #                           nn.Linear(opt.gh_dim, opt.X_dim),
        #                           nn.Sigmoid())
        self.main = nn.Sequential(nn.Linear(opt.Z_dim, 400),
                                  nn.ReLU(True),
                                  nn.Linear(400, 400),
                                  nn.ReLU(True),
                                  nn.Linear(400, 400),
                                  nn.ReLU(True),
                                  nn.Linear(400, opt.X_dim),
                                  nn.Sigmoid())

    def forward(self, z, c):
        # in_vec = torch.cat([z, c], dim=1)
        output = self.main(z)
        return output


def train():
    dataset = DATA_LOADER(opt)
    opt.C_dim = dataset.att_dim
    opt.X_dim = dataset.feature_dim
    opt.Z_dim = opt.latent_dim
    opt.y_dim = dataset.ntrain_class

    # Continual Learning dataset split
    taskset = dataset.ntrain_class // opt.task_split_num
    task_boundary = [ii for ii in range(0, dataset.ntrain_class+1, taskset)]
    start_idx = task_boundary.pop(0)

    result_knn = Result()

    netG = ConditionalGenerator(opt).to(device)
    netG.apply(weights_init)
    print(netG)
    out_dir = f'out/{opt.dataset}/nreplay-{opt.nSample_replay}_sigma-{opt.sigma}_langevin_s-{opt.langevin_s}_' \
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

    start_step = 0
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            log_dir = checkpoint['log_dir']
            cl_acc_dir = checkpoint['cl_acc_dir']
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    replay_mem = None

    # pi_c = torch.ones(dataset.ntrain_class,).float() / dataset.ntrain_class
    # pi_c.requires_grad_()
    mu_c = torch.zeros(dataset.ntrain_class, opt.Z_dim).normal_(0, 3).float()
    # mu_c.requires_grad_()
    log_sigma2_c = torch.zeros(dataset.ntrain_class, opt.Z_dim).float()
    # log_sigma2_c.requires_grad_()

    replay_stats = None
    all_task_dataloader = {}
    all_task_testdata = {}
    log_print(f"Task Boundary: {task_boundary} \n", log_dir)
    for task_idx, tb in enumerate(task_boundary):

        task_mask_1 = dataset.train_label >= start_idx
        task_mask_2 = dataset.train_label < tb
        task_mask = task_mask_1 * task_mask_2
        if replay_mem is not None:
            train_label = torch.cat([replay_mem[1], dataset.train_label[task_mask]], dim=0)
            train_feas = torch.cat([replay_mem[0], dataset.train_feature[task_mask]], dim=0)
        else:
            train_label = dataset.train_label[task_mask]
            train_feas = dataset.train_feature[task_mask]

        task_dataloader = FeatDataLayer(train_label.numpy(), train_feas.numpy(), opt.batchsize)
        all_task_dataloader[task_idx] = task_dataloader

        task_mask_1 = dataset.test_seen_label >= start_idx
        task_mask_2 = dataset.test_seen_label < tb
        task_mask = task_mask_1 * task_mask_2
        all_task_testdata[task_idx] = (dataset.test_seen_feature[task_mask], dataset.test_seen_label[task_mask])
        start_idx = tb

        if opt.resume and task_idx < 1:
            if os.path.isfile(opt.resume):
                # print("=> loading checkpoint '{}'".format(opt.resume))
                # checkpoint = torch.load(opt.resume)
                netG.load_state_dict(checkpoint['state_dict_G'])
                replay_stats = (checkpoint['replay_mem0'],
                                checkpoint['replay_mem1'],
                                checkpoint['replay_mem2'])
                mus = checkpoint["mu"]
                logsigma = checkpoint["logsigma"]
                mu_c[:mus.size(0)] = mus.data
                log_sigma2_c[:logsigma.size(0)] = logsigma.data
                replay_mem = (replay_stats[0], replay_stats[1])
            else:
                print("=> no checkpoint found at '{}'".format(opt.resume))
            print("========= Skipping Task 1 =========\n")
            continue

        # train_z = torch.randn(task_dataloader.num_obs, opt.Z_dim).float().to(device)
        if task_idx < 1:
            print("Pre-training Initialization")
            mus, logsigma = pre_train_gmm(opt.X_dim, opt.Z_dim, opt.gh_dim, task_dataloader,
                                          len(np.unique(task_dataloader.label)), 5, device, netG)
            print(f"Pre-train initialization complete.")
            mu_c[:mus.size(0)] = mus.data
            log_sigma2_c[:logsigma.size(0)] = logsigma.data
        else:
            # pass
            boundary = [0,2,4,6,8][task_idx]
            mus = loc_init_mu(mu_c[:boundary], mu_c[boundary:boundary+2])
            mu_c[boundary:boundary+2] = mus.data

        task_init_stat = get_class_stats_imp(mu_c, log_sigma2_c, task_dataloader.label)
        train_z = loc_init_z(task_init_stat, task_dataloader.label, opt.Z_dim, task_dataloader)
        if task_idx > 0 and replay_stats is not None:
            prev_z_size = replay_stats[-1].size(0)
            train_z[:prev_z_size] = replay_stats[-1]
        train_z = train_z.float().to(device)

        print(f"============ Task {task_idx+1} ============")
        print(f"Task Labels: {np.unique(task_dataloader.label)}")
        print(f"Task training shape: {task_dataloader.feat_data.shape}")
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        task_mu = mu_c[np.unique(task_dataloader.label)].to(device)
        task_mu.requires_grad_()
        task_logsigma = log_sigma2_c[np.unique(task_dataloader.label)].to(device)
        task_logsigma.requires_grad_()
        optimizerG.add_param_group({"params": [task_mu, task_logsigma]})
        total_niter = int(task_dataloader.num_obs/opt.batchsize) * opt.nepoch
        for it in range(start_step, total_niter+1):
            blobs = task_dataloader.forward()
            feat_data = blobs['data']  # image data
            labels = blobs['labels'].astype(int)  # class labels
            idx = blobs['idx'].astype(int)
            # subset_idx_mask = idx < prev_z_size
            # subset_idx = idx[subset_idx_mask]
            C = np.array([dataset.train_att[i, :] for i in labels])
            C = torch.from_numpy(C.astype('float32')).to(device)
            X = torch.from_numpy(feat_data).to(device)
            Z = train_z[idx]
            Z.requires_grad_()
            optimizer_z = torch.optim.Adam([Z, task_mu, task_logsigma], lr=opt.lr, weight_decay=opt.weight_decay)
            scheduler_z = optim.lr_scheduler.StepLR(optimizer_z, step_size=2, gamma=0.9)
            # Alternate update weights w and infer latent_batch z
            iter_loss = 0
            for em_step in range(1):  # EM_STEP
                # infer z
                for _ in range(opt.langevin_step):
                    U_tau = torch.FloatTensor(Z.shape).normal_(0, opt.sigma_U).to(device)
                    pred = netG(Z, C)
                    recon_loss = get_recon_loss(pred, X, opt.sigma)
                    entropy_loss, class_pred_prob = get_entropy_loss(Z, task_mu, task_logsigma,
                                                                     torch.from_numpy(labels).to(device))
                    prior_loss = get_prior_loss(Z, task_mu, task_logsigma, class_pred_prob, True)
                    loss = recon_loss + prior_loss + entropy_loss
                    # if task_idx > 0:
                    #     reg_loss = latent_reg(Z[subset_idx], replay_stats[-1][subset_idx])
                    #     loss += reg_loss
                    # loss = recon_loss + prior_loss
                    loss = (task_dataloader.num_obs / Z.size(0)) * loss
                    scaled_loss = (opt.langevin_s ** 2) * loss * 0.5
                    optimizer_z.zero_grad()
                    scaled_loss.backward()
                    # torch.nn.utils.clip_grad_norm_([Z], 5.)
                    # torch.nn.utils.clip_grad_norm_([task_mu, task_logsigma], 5.)
                    optimizer_z.step()
                    Z.data += opt.langevin_s * U_tau
                    # iter_loss += loss.detach()
                    scheduler_z.step()
                # TODO: regularize Z

                # update w
                for _ in range(1):
                    pred = netG(Z, C)
                    recon_loss = get_recon_loss(pred, X, opt.sigma)
                    entropy_loss, class_pred_prob = get_entropy_loss(Z, task_mu, task_logsigma,
                                                                     torch.from_numpy(labels).to(device))
                    prior_loss = get_prior_loss(Z, task_mu, task_logsigma, class_pred_prob, True)
                    loss = recon_loss + prior_loss + entropy_loss
                    # loss = recon_loss + prior_loss
                    scaled_loss = (task_dataloader.num_obs / Z.size(0)) * loss
                    optimizerG.zero_grad()
                    scaled_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(netG.parameters(), 5.)
                    optimizerG.step()
                    iter_loss += loss.detach()
            # update Z
            train_z[idx] = Z.data

            if it % opt.disp_interval == 0 and it:
                log_text = f'Iter-[{it}/{total_niter}]; loss: {iter_loss :.3f}'
                log_print(log_text, log_dir)

            if it % opt.evl_interval == 0 and it:
                netG.eval()
                task_stats = get_class_stats(train_z, task_dataloader.label)
                acc = eval_model_cls(netG, all_task_testdata[task_idx][0], all_task_testdata[task_idx][1],
                                     128, task_stats, opt.Z_dim, opt.sigma, device)
                result_knn.update(it, acc)
                log_print(f"{opt.Knn}nn Classifer: ", log_dir)
                log_print(f"Accuracy is {acc :.2f}%, Best_acc [{result_knn.best_acc :.2f}% "
                          f"| Iter-{result_knn.best_iter}]", log_dir)

                # if result_knn.save_model:
                #     files2remove = glob.glob(out_dir + '/Best_model_Knn_*')
                #     for _i in files2remove:
                #         os.remove(_i)
                #     save_model(it, netG, train_z, opt.manualSeed, log_text,
                #                out_dir + '/Best_model_Knn_Acc_{:.2f}.tar'.format(result_knn.acc_list[-1]))
                netG.train()

        mu_c[np.unique(task_dataloader.label)] = task_mu.data.to("cpu")
        log_sigma2_c[np.unique(task_dataloader.label)] = task_logsigma.data.to("cpu")
        print(f"============ Task {task_idx + 1} CL Evaluation ============")
        netG.eval()
        task_stats = get_class_stats(train_z, task_dataloader.label)
        test_acc = []
        for atask in range(task_idx + 1):
            print(np.unique(all_task_testdata[atask][1]))
            acc = eval_model_cls(netG, all_task_testdata[atask][0], all_task_testdata[atask][1],
                                 128, task_stats, opt.Z_dim, opt.sigma, device)
            test_acc.append(acc)
        result_knn.update_task_acc(test_acc)
        print(f"CL Task Accuracy: {test_acc}")
        print(f"CL Avg Accuracy: {np.mean(test_acc)}")
        if task_idx > 0:
            forget_rate = (result_knn.task_acc[-1][0] - result_knn.task_acc[0][0])
            forget_rate /= result_knn.task_acc[0][0]
            print(f"CL Forgetting: {np.abs(forget_rate)*100: .4f}%")

        if opt.nSample_replay > 0:
            print(f"============ Generate replay ============")
            # TODO: Improve generated features quality?
            # TODO: Stabilize sampling variation from latent distribution.
            # task_stats = get_class_stats_imp(task_mu, task_logsigma, task_dataloader.label)
            # torch.save({"latent_z": train_z.detach().to("cpu"),
            #             "labels": task_dataloader.label}, f"task{task_idx}_z.tar")
            replay_stats = synthesize_features(netG,
                                               task_dataloader.label, task_dataloader.label,
                                               dataset.train_att, task_stats,
                                               opt.nSample_replay, opt.X_dim, opt.Z_dim, True)
            # replay_stats = samp_features(task_dataloader.label, task_dataloader.feat_data, train_z,
            #                              opt.nSample_replay)
            log_print(f"Replay on: {replay_stats[0].shape}", log_dir)
            log_print(f"Replay labels: {np.unique(replay_stats[1])}", log_dir)
            replay_mem = (replay_stats[0], replay_stats[1])
        if opt.save_task and task_idx < 1:
            save_model(it, netG, replay_stats, opt.manualSeed, log_dir, cl_acc_dir, task_mu, task_logsigma,
                       f"{out_dir}/train_task_{task_idx+1}.tar")
            print(f'Save model to {out_dir}/train_task_{task_idx+1}.tar')
        print(f"============ End of task {task_idx + 1} ============\n")
        netG.train()
    result_knn.log_results(cl_acc_dir)


def log_print(s, log, print_str=True):
    if print_str:
        print(s)
    with open(log, 'a') as f:
        f.write(s + '\n')


def log_list(res_arr, log):
    num_item = len(res_arr)
    with open(log, "a") as f:
        for ii, res in enumerate(res_arr):
            if ii == num_item-1:
                f.write(f"{res}\n")
            else:
                f.write(f"{res},")


def get_recon_loss(pred, x, sigma):
    recon_loss = 1/(2*sigma**2) * torch.pow(x - pred, 2).sum()
    return recon_loss


def get_prior_loss(z, mus, log_sigma, class_prob, logits=False):
    pi_const = np.log(2.0 * np.pi)
    labels = class_prob
    if logits:
        labels = torch.argmax(class_prob, dim=1).detach()
    var_term = log_sigma[labels]
    dist_term = torch.pow(z - mus[labels], 2)/torch.exp(log_sigma[labels])
    dist_term = torch.sum(dist_term + var_term)
    loss = 0.5 * (pi_const + dist_term)
    return loss


def get_entropy_loss(z, mus, log_sigma, labels):
    loss_fcn = nn.CrossEntropyLoss(reduction='sum')
    num_mu = mus.size(0)
    latent_dim = mus.size(1)
    diag_mat = torch.eye(latent_dim).unsqueeze(0).to(device) * torch.exp(log_sigma).unsqueeze(1)
    log_pdf = dist.MultivariateNormal(mus, diag_mat).log_prob(z.unsqueeze(1).repeat(1,num_mu,1))
    yita_c = log_pdf
    loss = loss_fcn(yita_c, labels)
    return loss, yita_c


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


def get_class_stats(latent, labels):
    assert latent.size(0) == labels.shape[0]
    unique_label = np.unique(labels)
    train_stats = {}
    # train_loc = torch.zeros(len(unique_label), latent.size(1))
    for ii, label in enumerate(unique_label):
        mask = labels == label
        z_samp = latent[mask]
        var, loc = torch.var_mean(z_samp, dim=0)
        train_stats[label] = (loc, var)
    return train_stats


def get_class_stats_imp(locs, sigmas, labels):
    unique_label = np.unique(labels)
    train_stats = {}
    for ii, label in enumerate(unique_label):
        train_stats[label] = (locs[label].detach(), sigmas[label].detach())
    return train_stats


def loc_init_z(locs, prev_labels, latent_dim, dataloader):
    unique_task_labels = np.unique(prev_labels)
    train_z = torch.randn(dataloader.num_obs, latent_dim)
    for ii, task_label in enumerate(unique_task_labels):
        mask = dataloader.label == task_label
        # train_z[mask] += locs[ii].unsqueeze(0).repeat(np.sum(mask), 1)
        diag_mat = torch.exp(locs[task_label][1])*torch.eye(latent_dim)
        train_z[mask] = dist.MultivariateNormal(locs[task_label][0].to("cpu"),
                                                diag_mat.to("cpu")).sample([np.sum(mask)])
        # train_z[mask] = dist.MultivariateNormal(locs[task_label][0].to("cpu"),
        #                                         torch.eye(latent_dim)).sample([np.sum(mask)])
    return train_z


def loc_init_mu(prev_mu, cur_mu):
    ata = torch.matmul(prev_mu, prev_mu.t())
    inv_ata = torch.inverse(ata)
    proj_mat = torch.matmul(prev_mu.t(), torch.matmul(inv_ata, prev_mu))
    new_mu = cur_mu - torch.matmul(proj_mat, cur_mu.t()).t()
    return new_mu


def synthesize_features(netG, test_labels, trained_labels, attr, class_stats, n_samples, x_dim, z_dim, replay=False):
    unique_test_labels = np.unique(test_labels)
    unique_trained_labels = np.unique(trained_labels)
    nclass = len(unique_test_labels)
    gen_feat = torch.FloatTensor(nclass * n_samples, x_dim)
    gen_label = np.zeros([0])
    fix_z = []
    with torch.no_grad():
        for ii in range(nclass):
            test_label = unique_test_labels[ii]
            test_feat = np.tile(attr[test_label].astype('float32'), (n_samples, 1))
            test_feat = torch.from_numpy(test_feat).to(device)
            z = torch.randn(n_samples, z_dim)
            if test_label in class_stats.keys():
                # z += class_locs[test_label].unsqueeze(0).repeat(n_samples, 1)
                if replay:
                    z = dist.Normal(class_stats[test_label][0],
                                    torch.ones(z_dim).to(device)).sample([n_samples])
                else:
                    z = dist.Normal(class_stats[test_label][0],
                                    torch.exp(class_stats[test_label][1])).sample([n_samples])
            fix_z.append(z.detach())
            G_sample = netG(z.to(device), test_feat)
            gen_feat[ii*n_samples:(ii+1)*n_samples] = G_sample
            gen_label = np.hstack((gen_label, np.ones([n_samples])*test_label))
    fix_z = torch.cat(fix_z, dim=0)
    return gen_feat, torch.from_numpy(gen_label.astype(int)), fix_z


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
        start_idx = index_boundary[ii-1]
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
                res_arr = each_res + [0.0]*(num_item-len(each_res))
            else:
                res_arr = each_res
            log_list(res_arr, log)


def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.normal_(m.weight.data,  mean=0, std=0.02)
        init.constant_(m.bias, 0.0)


if __name__ == "__main__":
    for _ in range(opt.repeat):
        train()

