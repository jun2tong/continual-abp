import pdb
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
# import torch.nn.init as init
# import torchvision.utils as vutils

# import matplotlib.pyplot as plt
# import glob
import json
import argparse
import os
import random
import numpy as np
from time import gmtime, strftime

from models.gen_model import FeaturesGenerator
from classifier import eval_model, train_classifier
from dataset_GBU import FeatDataLayer, DATA_LOADER


parser = argparse.ArgumentParser()
parser.add_argument('--repeat', type=int, default=1, help='number of repeats for experiment')
parser.add_argument('--dataset', default='imagenet', help='dataset: CUB, AWA1, AWA2, SUN')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--image_embedding', default='resnet18_imagenet100_feas', type=str)
parser.add_argument('--task_split_num', type=int, default=20, help='number of task split')
parser.add_argument('--first_split', type=int, default=500, help='first split index')

parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--optimizer', default='Adam', help='optimizer: Adam or SGD')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate to train generator')
parser.add_argument('--z_lr', type=float, default=0.0002, help='learning rate to train latent')

parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--weight_decay', type=float, default=0., help='weight_decay')
parser.add_argument('--batchsize', type=int, default=1024, help='input batch size')
parser.add_argument('--nSample', type=int, default=800, help='number features to generate per class')

parser.add_argument('--resume', type=str, help='the model to resume')
parser.add_argument('--disp_interval', type=int, default=5000)
parser.add_argument('--save_task', type=int, default=0)
parser.add_argument('--evl_interval', type=int, default=20000)
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--latent_dim', type=int, default=100, help='dimension of latent z')
parser.add_argument('--gh_dim',     type=int, default=1024, help='dimension of hidden layer in generator')
parser.add_argument('--latent_var', type=float, default=1, help='variance of prior distribution z')

parser.add_argument('--sigma', type=float, default=0.1, help='variance of random noise')
parser.add_argument('--sigma_U', type=float, default=1, help='variance of U_tau')
parser.add_argument('--langevin_s', type=float, default=0.3, help='step size in langevin sampling')
parser.add_argument('--langevin_step', type=int, default=20, help='langevin step in each iteration')

parser.add_argument('--Knn', type=int, default=0, help='K value')
parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
parser.add_argument('--pca', type=int, default=0, help='PCA dimensions')
opt = parser.parse_args()


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True
device = f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu"

print('Running parameters:')
print(json.dumps(vars(opt), indent=4, separators=(',', ': ')))
print(device)


def train():
    dataset = DATA_LOADER(opt)
    opt.X_dim = dataset.feature_dim
    opt.Z_dim = opt.latent_dim
    num_k = 1000
    task_boundary = [ii for ii in range(500, dataset.ntrain_class+1, opt.task_split_num)]
    start_idx = 0
    result_lin = Result()

    netG = FeaturesGenerator(opt.Z_dim, num_k, opt.X_dim, opt.gh_dim).to(device)
    netG.apply(weights_init)
    print(netG)

    out_dir = f'out/{opt.dataset}/S{opt.task_split_num}-nepoch{opt.nepoch}-nSample{opt.nSample}'
    os.makedirs(out_dir, exist_ok=True)
    print("The output dictionary is {}".format(out_dir))

    it = 1
    while os.path.isfile(f"{out_dir}/log_it{it:02d}.txt"):
        it += 1
    log_dir = f"{out_dir}/log_it{it:02d}.txt"
    cl_acc_dir = f"{out_dir}/lin_acc_it{it:02d}.txt"
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
            train_latent = torch.zeros(train_label.size(0), opt.Z_dim)
            train_latent.data[:replay_mem[-1].size(0)] = replay_mem[-1]
        else:
            train_label = dataset.train_label[task_mask]
            train_feas = dataset.train_feature[task_mask]
            train_latent = torch.zeros(train_label.size(0), opt.Z_dim).float()

        log_print(f"train X: {train_feas.shape}", log_dir)
        log_print(f"train Y : {train_label.shape}", log_dir)

        task_dataloader = FeatDataLayer(train_label.numpy(), train_feas.numpy(),
                                        train_latent.numpy(), opt.batchsize)
        all_task_dataloader[task_idx] = task_dataloader

        task_mask_1 = dataset.test_seen_label >= start_idx
        task_mask_2 = dataset.test_seen_label < tb
        task_mask = task_mask_1 * task_mask_2
        all_task_testdata[task_idx] = (dataset.test_seen_feature[task_mask], dataset.test_seen_label[task_mask])
        start_idx = tb

        if opt.optimizer == "Adam":
            optimizer_g = torch.optim.Adam(netG.parameters(), lr=opt.lr,
                                           weight_decay=opt.weight_decay, betas=(0.5, 0.999))
        else:
            optimizer_g = torch.optim.SGD(netG.parameters(), lr=opt.lr,
                                          momentum=opt.g_momentum,
                                          weight_decay=opt.weight_decay)

        train_z = torch.randn(task_dataloader.num_obs, opt.Z_dim)
        if task_idx > 0 and replay_mem is not None:
            prev_z_size = replay_mem[-1].size(0)
            train_z[:prev_z_size] = replay_mem[-1]
        train_z = train_z.float().to(device)

        print(f"============ Task {task_idx+1} ============")
        # print(f"Task Labels: {np.unique(task_dataloader.label)}")
        print(f"Task training shape: {task_dataloader.feat_data.shape}")
        total_niter = int(task_dataloader.num_obs/opt.batchsize) * opt.nepoch
        if task_idx < 1:
            total_niter = int(task_dataloader.num_obs/opt.batchsize) * 120
        # total_niter = opt.nepoch
        print(f"{int(task_dataloader.num_obs/opt.batchsize)} iter per epoch")
        n_active = len(np.unique(task_dataloader.label))
        best_acc = 0
        for it in range(start_step, total_niter+1):
            blobs = task_dataloader.forward()
            feat_data = blobs['data']  # image data
            y_mb = torch.from_numpy(blobs['labels'].astype(int))  # class labels
            idx = blobs['idx'].astype(int)
            x_mb = torch.from_numpy(feat_data).to(device)
            z_mb = train_z[idx]
            z_mb.requires_grad_()
            prev_z = torch.from_numpy(blobs["latent_z"]).to(device)

            optimizer_z = torch.optim.Adam([z_mb], lr=opt.lr, weight_decay=opt.weight_decay, betas=(0.5, 0.999))

            # Alternate update weights w and infer latent_batch z
            batch_loss = 0
            for em_step in range(1):  # EM_STEP
                optimizer_g.zero_grad()
                one_hot_y = torch.eye(num_k)[y_mb]
                recon_x = netG(z_mb, one_hot_y.to(device))

                recon_loss = get_recon_loss(recon_x, x_mb, opt.sigma)  # Reconstruction Loss

                prior_loss = get_prior_loss(z_mb, prev_z)

                gloss = recon_loss + prior_loss
                gloss /= x_mb.size(0)
                gloss.backward()
                optimizer_g.step()
                srmc_loss = 0
                for ls_step in range(opt.langevin_step):
                    optimizer_z.zero_grad()
                    u_tau = torch.randn(z_mb.size(0), opt.Z_dim).float().to(device)

                    one_hot_y = torch.eye(num_k)[y_mb]
                    recon_x = netG(z_mb, one_hot_y.to(device))
                    recon_loss = get_recon_loss(recon_x, x_mb, opt.sigma)
                    prior_loss = get_prior_loss(z_mb, prev_z)

                    loss = recon_loss + prior_loss
                    loss /= x_mb.size(0)
                    loss = loss * (opt.langevin_s*opt.langevin_s)/2
                    loss.backward()
                    optimizer_z.step()
                    if ls_step < 5 and it < int(task_dataloader.num_obs/opt.batchsize):
                        z_mb.data += u_tau * opt.sigma
                    srmc_loss += loss.detach()

                train_z[idx,] = z_mb.data
                batch_loss += (srmc_loss / opt.langevin_step) + gloss.detach()

            if it == int(total_niter*0.6):
                optimizer_g.param_groups[0]["lr"] = 0.0003

            if it == int(total_niter*0.8):
                optimizer_g.param_groups[0]["lr"] = 0.0002

            if (it+1) % int(task_dataloader.num_obs/opt.batchsize) == 0:
                netG.eval()
                # n_active = len(np.unique(task_dataloader.label))
                # replay_mem = synthesize_features(netG, 300, n_active, num_k, opt.X_dim, opt.Z_dim)
                replay_mem = synthesize_features(netG, opt.nSample, train_z, task_dataloader.label, num_k, opt.X_dim, opt.Z_dim)
                # lin_cls = train_classifier(torch.from_numpy(task_dataloader.feat_data), torch.from_numpy(task_dataloader.label).long(),
                #                            opt.classifier_lr, device, n_active)
                lin_cls = train_classifier(replay_mem[0], replay_mem[1],
                                           opt.classifier_lr, device, n_active)
                # lin_cls = train_classifier(torch.from_numpy(task_dataloader.feat_data), torch.from_numpy(task_dataloader.label).long(),
                #                            opt.classifier_lr, device, n_active)
                eval_acc = eval_model(lin_cls, all_task_testdata[task_idx][0], all_task_testdata[task_idx][1], device)
                if eval_acc > best_acc:
                    best_acc = eval_acc
                    log_text = f'Eval-[{it}/{total_niter}]; loss: {batch_loss :.4f}; accuracy: {eval_acc: .4f}'
                    torch.save({"netG": netG.state_dict()}, "model.pth")
                    log_print(log_text, log_dir)
                netG.train()

            if it % opt.disp_interval == 0 and it:
                log_text = f'Iter-[{it}/{total_niter}]; loss: {batch_loss :.4f}; acc: {eval_acc}; lr: {optimizer_g.param_groups[0]["lr"]}'
                log_print(log_text, log_dir)

        print(f"============ Task {task_idx + 1} CL Evaluation ============")
        netG.eval()
        test_acc = []
        n_active = len(np.unique(task_dataloader.label))
        netG.load_state_dict(torch.load("model.pth")["netG"])
        # replay_mem = synthesize_features(netG, opt.nSample, n_active, num_k, opt.X_dim, opt.Z_dim)
        replay_mem = synthesize_features(netG, opt.nSample, train_z, task_dataloader.label, num_k, opt.X_dim, opt.Z_dim)
        lin_cls = train_classifier(replay_mem[0], replay_mem[1],
                                   opt.classifier_lr, device, n_active)
        # name_c = 1
        # while os.path.isfile(f"{out_dir}/state-task{task_idx:02d}-{name_c:02d}.tar"):
        #     name_c += 1
        for atask in range(task_idx + 1):
            eval_acc = eval_model(lin_cls, all_task_testdata[atask][0], all_task_testdata[atask][1], device)
            test_acc.append(eval_acc)
        result_lin.update_task_acc(test_acc)
        print(f"CL Task Accuracy: {test_acc}")
        print(f"CL Avg Accuracy: {np.mean(test_acc):.4f}")
        print(f"============ End of task {task_idx + 1} ============\n")
        netG.train()
    result_lin.log_results(cl_acc_dir)


def log_print(s, log, print_str=True):
    if print_str:
        print(s)


def log_list(res_arr, log):
    num_item = len(res_arr)
    with open(log, "a") as f:
        for ii, res in enumerate(res_arr):
            if ii == num_item-1:
                f.write(f"{res}\n")
            else:
                f.write(f"{res},")


def get_recon_loss(pred, x, sigma):
    # recon_loss = nn.functional.binary_cross_entropy(pred, x, reduction='sum')
    recon_loss = nn.functional.mse_loss(pred, x, reduction='sum') / (2*sigma*sigma)
    # recon_loss = 1/(2*sigma*sigma) * torch.pow(x - pred, 2).sum()
    return recon_loss


def get_prior_loss(z, prev_rep):
    log_pdf = nn.functional.mse_loss(z,prev_rep,reduction='sum') * 0.5
    # log_pdf = 0.5 * torch.sum(torch.pow(z - prev_rep, 2))
    return log_pdf


def get_prior_loss_mm(net, z, x, num_active, num_k=100):
    all_dist = []
    for ii in range(num_active):
        one_hot_y = torch.eye(num_k)[ii]
        one_hot_y = one_hot_y.repeat(x.size(0),1).to(device)
        recon_x = net(z, one_hot_y)
        dist = -1.*torch.sum(torch.pow(recon_x - x, 2), dim=-1)
        all_dist.append(dist.unsqueeze(1))
    log_pdf = torch.cat(all_dist, dim=1)
    return log_pdf


def get_entropy_loss(logits, probs):
    log_q = torch.log_softmax(logits, dim=1)
    return torch.sum(-torch.sum(probs * log_q, dim=-1))


def save_state(replay_mem, fout):
    torch.save({
        'replay_mem0': replay_mem[0].cpu().detach(),
        'replay_mem1': replay_mem[1].cpu().detach(),
        'replay_mem2': replay_mem[2].cpu().detach(),
    }, fout)


def save_model(it, netG, latent_state, replay_mem, random_seed, log, acc_log, fout):
    torch.save({
        'it': it + 1,
        'state_dict_G': netG.state_dict(),
        'train_z_state': latent_state.cpu().detach(),
        'replay_mem0': replay_mem[0].cpu().detach(),
        'replay_mem1': replay_mem[1].cpu().detach(),
        'replay_mem2': replay_mem[2].cpu().detach(),
        'random_seed': random_seed,
        # 'mus': mus.cpu().detach(),
        # 'logsigma': logsigma.cpu().detach(),
        'log_dir': log,
        'cl_acc_dir': acc_log
    }, fout)


def synthesize_features(netG, n_samples, latent_vars, labels, num_k, x_dim, latent_dim):
    n_active_comp = len(np.unique(labels))
    gen_feat = torch.empty(n_active_comp*n_samples, x_dim).float()
    gen_label = np.zeros([0])
    replay_z = []
    with torch.no_grad():
        # z = torch.randn(n_samples, latent_dim).to(device)
        for ii in range(n_active_comp):
            one_hot_y = torch.eye(num_k)[ii]
            one_hot_y = one_hot_y.repeat(n_samples, 1)
            mask = labels==ii
            num_lab = np.sum(mask)
            if num_lab < n_samples:
                choice_idx = np.random.choice(np.arange(num_lab), size=(n_samples), replace=True)
            else:
                choice_idx = np.random.choice(np.arange(num_lab), size=(n_samples), replace=False)
            z = latent_vars[mask][choice_idx]
            G_sample = netG(z, one_hot_y.to(device))
            gen_feat[ii * n_samples:(ii + 1) * n_samples] = G_sample
            gen_label = np.hstack((gen_label, np.ones([n_samples]) * ii))
            replay_z.append(z)

    gen_label = torch.from_numpy(gen_label).long()
    replay_z = torch.cat(replay_z, dim=0)
    return gen_feat, gen_label, replay_z


# def synthesize_features(netG, n_samples, n_active_comp, num_k, x_dim, latent_dim):
#     gen_feat = torch.empty(n_active_comp*n_samples, x_dim).float()
#     gen_label = np.zeros([0])
#     replay_z = []
#     with torch.no_grad():
#         # z = torch.randn(n_samples, latent_dim).to(device)
#         for ii in range(n_active_comp):
#             one_hot_y = torch.eye(num_k)[ii]
#             one_hot_y = one_hot_y.repeat(n_samples, 1)
#             z = torch.randn(n_samples, latent_dim).to(device)
#             G_sample = netG(z, one_hot_y.to(device))
#             gen_feat[ii * n_samples:(ii + 1) * n_samples] = G_sample
#             gen_label = np.hstack((gen_label, np.ones([n_samples]) * ii))
#             replay_z.append(z)

#     gen_label = torch.from_numpy(gen_label).long()
#     replay_z = torch.cat(replay_z, dim=0)
    # return gen_feat, gen_label, replay_z


def get_feats(feats, labels, latent_z, n_samples):
    unique_label = np.unique(labels)
    feas_mem = []
    lab_mem = []
    z_mem = []
    for alab in unique_label:
        mask = labels == alab
        feas = feats[mask]
        z = latent_z[mask]
        lab_group = labels[mask]
        if feas.shape[0] > n_samples:
            indices = np.random.choice(np.arange(feas.shape[0]), n_samples, replace=False)
            feas = feas[indices]
            z = z[indices]
            lab_group = lab_group[indices]

        feas_mem.append(feas)
        z_mem.append(z)
        lab_mem.append(lab_group)

    feas_mem = torch.cat(feas_mem, dim=0)
    lab_mem = torch.cat(lab_mem, dim=0)
    z_mem = torch.cat(z_mem, dim=0)
    return feas_mem, lab_mem, z_mem


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
    if classname == "Linear":
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='leaky_relu')


if __name__ == "__main__":
    for _ in range(opt.repeat):
        train()
