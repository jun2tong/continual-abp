import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

import glob
import json
import argparse
import os
import random
import numpy as np
from time import gmtime, strftime
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

import classifier
from dataset_GBU import FeatDataLayer, DATA_LOADER


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
parser.add_argument('--lr', type=float, default=0.002, help='learning rate to train generater')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
parser.add_argument('--nSample', type=int, default=300, help='number features to generate per class')
parser.add_argument('--nSample_replay', type=int, default=100, help='number features for replay')

parser.add_argument('--resume',  type=str, help='the model to resume')
parser.add_argument('--disp_interval', type=int, default=20)
parser.add_argument('--save_interval', type=int, default=10000)
parser.add_argument('--evl_interval',  type=int, default=100)
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--latent_dim', type=int, default=10, help='dimension of latent z')
parser.add_argument('--gh_dim',     type=int, default=4096, help='dimension of hidden layer in generator')
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
        self.main = nn.Sequential(nn.Linear(opt.Z_dim+opt.C_dim, opt.gh_dim),
                                  nn.LeakyReLU(0.2, True),
                                  nn.Linear(opt.gh_dim, opt.X_dim),
                                  nn.ReLU(True))

    def forward(self, z, c):
        in_vec = torch.cat([z, c], dim=1)
        output = self.main(in_vec)
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
    # out_dir = f'out/{opt.dataset}/nSample-{opt.nSample}_nZ-{opt.Z_dim}_sigma-{opt.sigma}_langevin_s-{opt.langevin_s}_' \
    #           f'step-{opt.langevin_step}'
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
            netG.load_state_dict(checkpoint['state_dict_G'])
            train_z = checkpoint['latent_z'].to(device)
            start_step = checkpoint['it']
            print(checkpoint['log'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    replay_mem = None
    task_stats = None
    task_locs = None
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

        # TODO: Think about z initialization
        train_z = torch.FloatTensor(task_dataloader.num_obs, opt.Z_dim).normal_(0, opt.latent_var)
        if task_stats is not None:
            task_locs = []
            for key, val in task_stats.items():
                task_locs.append(val[0].unsqueeze(0))
            task_locs = torch.cat(task_locs, dim=0)
            # train_z = loc_init_z(task_stats, replay_mem[1], opt.Z_dim, task_dataloader)

        train_z = train_z.to(device)

        print(f"============ Task {task_idx+1} ============")
        print(f"Task Labels: {np.unique(task_dataloader.label)}")
        print(f"Task training shape: {task_dataloader.feat_data.shape}")
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        total_niter = int(task_dataloader.num_obs/opt.batchsize) * opt.nepoch
        for it in range(start_step, total_niter+1):
            blobs = task_dataloader.forward()
            feat_data = blobs['data']  # image data
            labels = blobs['labels'].astype(int)  # class labels
            idx = blobs['idx'].astype(int)

            C = np.array([dataset.train_att[i, :] for i in labels])
            C = torch.from_numpy(C.astype('float32')).to(device)
            X = torch.from_numpy(feat_data).to(device)
            Z = nn.Parameter(train_z[idx]).to(device)
            optimizer_z = torch.optim.Adam([Z], lr=opt.lr, weight_decay=opt.weight_decay)

            # Alternatingly update weights w and infer latent_batch z
            iter_loss = 0
            for em_step in range(1):  # EM_STEP
                # infer z
                for _ in range(opt.langevin_step):
                    U_tau = torch.FloatTensor(Z.shape).normal_(0, opt.sigma_U).to(device)
                    pred = netG(Z, C)
                    if task_locs is not None:
                        loss = getloss(pred, X, Z, opt.sigma, task_locs, labels)
                    else:
                        loss = getloss(pred, X, Z, opt.sigma, None, labels)
                    scaled_loss = (task_dataloader.num_obs / Z.size(0)) * loss
                    scaled_loss = (opt.langevin_s ** 2) / 2 * scaled_loss
                    optimizer_z.zero_grad()
                    scaled_loss.backward()
                    # torch.nn.utils.clip_grad_norm_([Z], 5.)
                    optimizer_z.step()
                    Z.data += opt.langevin_s * U_tau
                    iter_loss += loss.detach()
                # update w
                for _ in range(1):
                    pred = netG(Z, C)
                    if task_locs is not None:
                        loss = getloss(pred, X, Z, opt.sigma, task_locs, labels)
                    else:
                        loss = getloss(pred, X, Z, opt.sigma, None, labels)
                    scaled_loss = (task_dataloader.num_obs/Z.size(0)) * loss
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
                tmp_task_stats = get_class_stats(train_z, task_dataloader.label)
                gen_feat, gen_label = synthesize_features(netG,
                                                          task_dataloader.label, task_dataloader.label,
                                                          dataset.train_att, tmp_task_stats,
                                                          opt.nSample, opt.X_dim, opt.Z_dim)

                """Knn Classification"""
                all_feas = gen_feat
                all_label = gen_label
                acc = eval_knn(all_feas.numpy(), all_label.numpy(),
                               task_dataloader.feat_data, task_dataloader.label,
                               opt.Knn)
                result_knn.update(it, acc)
                log_print(f"{opt.Knn}nn Classifer: ", log_dir)
                log_print(f"Accuracy is {acc :.2f}%, Best_acc [{result_knn.best_acc :.2f}% "
                          f"| Iter-{result_knn.best_iter}]", log_dir)

                # TODO: Use clustering prior
                if result_knn.save_model:
                    files2remove = glob.glob(out_dir + '/Best_model_Knn_*')
                    for _i in files2remove:
                        os.remove(_i)
                    save_model(it, netG, train_z, opt.manualSeed, log_text,
                               out_dir + '/Best_model_Knn_Acc_{:.2f}.tar'.format(result_knn.acc_list[-1]))

                netG.train()

            if it % opt.save_interval == 0 and it:
                save_model(it, netG, train_z, opt.manualSeed, log_text,
                           out_dir + '/Iter_{:d}.tar'.format(it))
                print('Save model to ' + out_dir + '/Iter_{:d}.tar'.format(it))
        print(f"============ Task {task_idx + 1} CL Evaluation ============")
        netG.eval()
        # Generate features using labels seen so far.
        task_stats = get_class_stats(train_z, task_dataloader.label)
        gen_feat, gen_label = synthesize_features(netG,
                                                  task_dataloader.label, task_dataloader.label,
                                                  dataset.train_att, task_stats,
                                                  opt.nSample, opt.X_dim, opt.Z_dim)
        test_acc = []
        for atask in range(task_idx + 1):
            print(np.unique(all_task_testdata[atask][1]))
            acc = eval_knn(gen_feat, gen_label,
                           all_task_testdata[atask][0], all_task_testdata[atask][1],
                           opt.Knn)
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
            task_stats = get_class_stats(train_z, task_dataloader.label)
            # torch.save({"latent_z": train_z.detach().to("cpu"),
            #             "labels": task_dataloader.label}, f"task{task_idx}_z.tar")
            replay_feat, replay_label = synthesize_features(netG,
                                                            task_dataloader.label, task_dataloader.label,
                                                            dataset.train_att, task_stats,
                                                            opt.nSample_replay, opt.X_dim, opt.Z_dim, True)
            # replay_feat, replay_label = samp_features(task_dataloader.label, task_dataloader.feat_data, 300)
            log_print(f"Replay on: {replay_feat.shape}", log_dir)
            log_print(f"Replay labels: {np.unique(replay_label)}", log_dir)
            replay_mem = (replay_feat, replay_label)
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


def getloss(pred, x, z, sigma, prior_locs, labels):
    # loss = 1/(2*sigma**2) * torch.pow(x - pred, 2).sum() + 1/2 * torch.pow(z, 2).sum()
    # loss /= x.size(0)
    pred_loss = 1/(2*sigma**2) * torch.pow(x - pred, 2).sum()
    prior_loss = get_prior_loss(z, prior_locs, labels)
    loss = pred_loss + prior_loss
    # loss *= scale_factor / x.size(0)
    return loss


def get_prior_loss(z, prior_locs, labels):
    if prior_locs is not None:
        prev_label_max = prior_locs.size(0)
        mask = labels < prev_label_max
        aval_label = labels[mask]
        avl_loss = 1/2*torch.pow(z[mask] - prior_locs[aval_label].to(device), 2).sum()
        unavl_loss = 1/2*torch.pow(z[~mask], 2).sum()
        loss = avl_loss + unavl_loss
    else:
        loss = 1/2 * torch.pow(z, 2).sum()
    return loss


def save_model(it, netG, train_z, random_seed, log, fout):
    torch.save({
        'it': it + 1,
        'state_dict_G': netG.state_dict(),
        'latent_z': train_z,
        'random_seed': random_seed,
        'log': log,
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


def loc_init_z(locs, prev_labels, latent_dim, dataloader):
    unique_task_labels = np.unique(prev_labels)
    train_z = torch.randn(dataloader.num_obs, latent_dim)
    for ii, task_label in enumerate(unique_task_labels):
        mask = dataloader.label == task_label
        # train_z[mask] += locs[ii].unsqueeze(0).repeat(np.sum(mask), 1)
        train_z[mask] = torch.distributions.Normal(locs[task_label][0].to("cpu"),
                                                   torch.sqrt(locs[task_label][1].to('cpu'))).sample([np.sum(mask)])
    return train_z


def synthesize_features(netG, test_labels, trained_labels, attr, class_stats, n_samples, x_dim, z_dim, replay=False):
    unique_test_labels = np.unique(test_labels)
    unique_trained_labels = np.unique(trained_labels)
    nclass = len(unique_test_labels)
    gen_feat = torch.FloatTensor(nclass * n_samples, x_dim)
    gen_label = np.zeros([0])
    with torch.no_grad():
        for ii in range(nclass):
            test_label = unique_test_labels[ii]
            test_feat = np.tile(attr[test_label].astype('float32'), (n_samples, 1))
            test_feat = torch.from_numpy(test_feat).to(device)
            z = torch.randn(n_samples, z_dim)
            if test_label in class_stats.keys():
                # z += class_locs[test_label].unsqueeze(0).repeat(n_samples, 1)
                if replay:
                    z = torch.distributions.Normal(class_stats[test_label][0],
                                                   torch.ones(z_dim).to(device)).sample([n_samples])
                else:
                    # z = torch.distributions.Normal(class_stats[test_label][0],
                    #                                torch.sqrt(class_stats[test_label][1])).sample([n_samples])
                    z = torch.distributions.Normal(class_stats[test_label][0],
                                                   torch.ones(z_dim).to(device)).sample([n_samples])
            G_sample = netG(z.to(device), test_feat)
            gen_feat[ii*n_samples:(ii+1)*n_samples] = G_sample
            gen_label = np.hstack((gen_label, np.ones([n_samples])*test_label))

    return gen_feat, torch.from_numpy(gen_label.astype(int))


def samp_features(trained_labels, feas, n_samples):
    unique_trained_labels = np.unique(trained_labels)
    nclass = len(unique_trained_labels)
    gen_feat = []
    gen_label = []
    for ii in range(nclass):
        label = unique_trained_labels[ii]
        mask = trained_labels == label
        subset_feas = feas[mask]
        if subset_feas.shape[0] < n_samples:
            subsamp_idx = np.random.choice(np.arange(np.sum(mask)), subset_feas.shape[0], replace=False)
        else:
            subsamp_idx = np.random.choice(np.arange(np.sum(mask)), n_samples, replace=False)
        gen_feat.append(torch.from_numpy(subset_feas[subsamp_idx]))
        gen_label.append(trained_labels[mask])

    gen_feat = torch.cat(gen_feat, dim=0)
    gen_label = np.concatenate(gen_label)
    return gen_feat, torch.from_numpy(gen_label.astype(int))


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

