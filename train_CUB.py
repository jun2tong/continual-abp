import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
# import torch.nn.init as init
import torch.distributions as dist
# import torchvision.utils as vutils

# import matplotlib.pyplot as plt
# import glob
import json
import argparse
import os
import random
import numpy as np
from time import gmtime, strftime

from gen_model import FeaturesGenerator, CUBGenerator
from classifier import eval_model, eval_knn
from dataset_GBU import FeatDataLayer, DATA_LOADER


parser = argparse.ArgumentParser()
parser.add_argument('--repeat', type=int, default=1, help='number of repeats for experiment')
parser.add_argument('--dataset', default='AWA1', help='dataset: CUB, AWA1, AWA2, SUN')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--image_embedding', default='res_feas_t1-cifar10', type=str)
parser.add_argument('--task_split_num', type=int, default=5, help='number of task split')
parser.add_argument('--first_split', type=int, default=50, help='first split index')
parser.add_argument('--gen_type', type=int, default=1, help='generator type')

parser.add_argument('--nepoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--optimizer', default='Adam', help='optimizer: Adam or SGD')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate to train generator')
parser.add_argument('--g_momentum', type=float, default=0.9, help='SGD momentum to train generator')
parser.add_argument('--g_gamma', type=float, default=0.8, help='training scheduler gamma for generator')
parser.add_argument('--g_step', type=int, default=500, help='number of steps for lr decay.')
parser.add_argument('--z_lr', type=float, default=0.0002, help='learning rate to train latent')
parser.add_argument('--z_step', type=int, default=5, help='number of steps for lr decay for latent.')
parser.add_argument('--z_gamma', type=float, default=0.8, help='training scheduler gamma for latent')

parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
parser.add_argument('--nSample', type=int, default=300, help='number features to generate per class')

parser.add_argument('--resume', type=str, help='the model to resume')
parser.add_argument('--disp_interval', type=int, default=200)
parser.add_argument('--save_task', type=int, default=0)
parser.add_argument('--evl_interval', type=int, default=1000)
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--latent_dim', type=int, default=50, help='dimension of latent z')
parser.add_argument('--gh_dim',     type=int, default=1024, help='dimension of hidden layer in generator')
parser.add_argument('--latent_var', type=float, default=1, help='variance of prior distribution z')

parser.add_argument('--sigma', type=float, default=0.1, help='variance of random noise')
parser.add_argument('--sigma_U', type=float, default=1, help='variance of U_tau')
parser.add_argument('--langevin_s', type=float, default=0.3, help='step size in langevin sampling')
parser.add_argument('--langevin_step', type=int, default=30, help='langevin step in each iteration')

parser.add_argument('--Knn', type=int, default=0, help='K value')
parser.add_argument('--pca', type=int, default=0, help='PCA dimensions')
parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
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
    num_k = dataset.att_dim
    # Continual Learning dataset split
    # taskset = dataset.ntrain_class // opt.task_split_num
    task_boundary = [ii for ii in range(opt.first_split, dataset.ntrain_class+1, opt.task_split_num)]
    # start_idx = task_boundary.pop(0)
    start_idx = 0

    result_knn = Result()
    if opt.gen_type == 1:
        # netG = FeaturesGenerator(opt.Z_dim, num_k, opt.X_dim, opt.gh_dim, 200).to(device)
        netG = FeaturesGenerator(opt.Z_dim, num_k, opt.X_dim, opt.gh_dim).to(device)
    else:
        # netG = CUBGenerator(opt.Z_dim, num_k, opt.X_dim, opt.gh_dim, 200).to(device)
        netG = CUBGenerator(opt.Z_dim, num_k, opt.X_dim, opt.gh_dim).to(device)
    netG.apply(weights_init)
    print(netG)

    out_dir = f'out/{opt.dataset}/{opt.image_embedding}-nepoch{opt.nepoch}-nSample{opt.nSample}-gentype{opt.gen_type}-A{opt.task_split_num}'
    os.makedirs(out_dir, exist_ok=True)
    print("The output dictionary is {}".format(out_dir))

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
        print(f"Task Labels: {np.unique(task_dataloader.label)}")
        print(f"Task training shape: {task_dataloader.feat_data.shape}")
        if task_idx > 0:
            total_niter = int(task_dataloader.num_obs/opt.batchsize) * opt.nepoch
        else:
            total_niter = int(task_dataloader.num_obs/opt.batchsize) * 200
        # total_niter = opt.nepoch
        n_active = len(np.unique(task_dataloader.label))
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
            # scheduler_z = optim.lr_scheduler.StepLR(optimizer_z, step_size=opt.z_step, gamma=opt.z_gamma)

            # Alternate update weights w and infer latent_batch z
            batch_loss = 0
            for em_step in range(1):  # EM_STEP
                optimizer_g.zero_grad()
                cls_att = dataset.attribute[y_mb]
                one_hot_y = torch.eye(200)[y_mb]
                recon_x = netG(z_mb, cls_att.to(device))
                recon_loss = get_recon_loss(recon_x, x_mb, opt.sigma)  # Reconstruction Loss

                prior_loss = get_prior_loss(z_mb, prev_z)

                gloss = recon_loss + prior_loss
                gloss /= x_mb.size(0)
                gloss.backward()
                # torch.nn.utils.clip_grad_norm_(netG.parameters(), 1)
                optimizer_g.step()
                srmc_loss = 0
                for ls_step in range(opt.langevin_step):
                    optimizer_z.zero_grad()
                    u_tau = torch.randn(z_mb.size(0), opt.Z_dim).float().to(device)

                    cls_att = dataset.attribute[y_mb]
                    one_hot_y = torch.eye(200)[y_mb]
                    recon_x = netG(z_mb, cls_att.to(device))
                    recon_loss = get_recon_loss(recon_x, x_mb, opt.sigma)
                    prior_loss = get_prior_loss(z_mb, prev_z)

                    loss = recon_loss + prior_loss
                    loss /= x_mb.size(0)
                    loss = loss * (opt.langevin_s*opt.langevin_s)/2

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_([z_mb], 1)
                    optimizer_z.step()
                    # TODO: Maybe implement metropolis hastings selection here?
                    if ls_step < 1 and it == int(task_dataloader.num_obs/opt.batchsize):
                        z_mb.data += u_tau * opt.langevin_s
                    srmc_loss += loss.detach()
                    # scheduler_z.step()

                train_z[idx,] = z_mb.data
                batch_loss += (srmc_loss / opt.langevin_step) + gloss.detach()
            # batch_loss /= 2.
            # scheduler_g.step()

            if it % opt.disp_interval == 0 and it:
                log_text = f'Iter-[{it}/{total_niter}]; loss: {batch_loss :.4f}'
                log_print(log_text, log_dir)

            if it % opt.evl_interval == 0 and it:
                netG.eval()
                n_active = len(np.unique(task_dataloader.label))
                replay_mem = synthesize_feature_test(netG, dataset.attribute, task_dataloader.label, opt)
                eval_acc = eval_model(replay_mem[0], replay_mem[1],
                                      all_task_testdata[task_idx][0], all_task_testdata[task_idx][1],
                                      opt.classifier_lr, device, n_active)

                log_text = f'Eval-[{it}/{total_niter}]; loss: {batch_loss :.4f}; accuracy: {eval_acc: .4f}'
                log_print(log_text, log_dir)
                netG.train()

        print(f"============ Task {task_idx + 1} CL Evaluation ============")
        netG.eval()
        test_acc = []
        knn_test_acc = []
        n_active = len(np.unique(task_dataloader.label))
        replay_mem = synthesize_feature_test(netG, dataset.attribute, task_dataloader.label, opt)
        for atask in range(task_idx + 1):
            print(np.unique(all_task_testdata[atask][1]))
            eval_acc = eval_model(replay_mem[0], replay_mem[1],
                                  all_task_testdata[atask][0], all_task_testdata[atask][1],
                                  opt.classifier_lr, device, n_active)
            knn_acc = eval_knn(replay_mem[0].numpy(), replay_mem[1].numpy(), 
                               all_task_testdata[atask][0].numpy(), all_task_testdata[atask][1].numpy(), 20)
            test_acc.append(eval_acc)
            knn_test_acc.append(knn_acc)
        result_knn.update_task_acc(test_acc)
        print(f"CL Task Accuracy: {test_acc}; Knn {knn_test_acc}")
        print(f"CL Avg Accuracy: {np.mean(test_acc):.4f}; Knn Avg Acc: {np.mean(knn_test_acc):.4f}")
        # zsl_mem = synthesize_feature_test(netG, dataset.attribute, dataset.unseenclasses, opt)
        # eval_acc = eval_model(zsl_mem[0], zsl_mem[1],
        #                       dataset.test_unseen_feature, dataset.test_unseen_label,
        #                       opt.classifier_lr, device, 200)
        # print(f"ZSL Eval: {eval_acc}")
        if task_idx > 0:
            forget_rate = (result_knn.task_acc[-1][0] - result_knn.task_acc[0][0])
            forget_rate /= result_knn.task_acc[0][0]
            print(f"CL Forgetting: {np.abs(forget_rate)*100: .4f}%")

        if opt.save_task and task_idx < 1:
            save_model(it, netG, train_z, replay_mem, opt.manualSeed, log_dir, cl_acc_dir,
                       f"{out_dir}/train_task_{task_idx+1}.tar")
            print(f'Save model to {out_dir}/train_task_{task_idx+1}.tar')
        print(f"============ End of task {task_idx + 1} ============\n")
        netG.train()
    result_knn.log_results(cl_acc_dir)


def log_print(s, log, print_str=True):
    if print_str:
        print(s)
    # with open(log, 'a') as f:
    #     f.write(s + '\n')


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


def synthesize_feature_test(netG, cls_attr, cls_label, opt):
    unique_labels = np.unique(cls_label)
    ntest_class = unique_labels.shape[0]
    gen_feat = torch.FloatTensor(ntest_class * opt.nSample, opt.X_dim)
    gen_label = np.zeros([0])
    replay_z = []
    with torch.no_grad():
        for idx, alab in enumerate(unique_labels):
            text_feat = cls_attr[alab].unsqueeze(0).repeat(opt.nSample,1)
            # text_feat = np.tile(cls_attr[i].astype('float32'), (opt.nSample, 1))
            text_feat = text_feat.cuda()
            z = torch.randn(opt.nSample, opt.Z_dim).cuda()
            one_hot_y = torch.eye(200)[idx]
            one_hot_y = one_hot_y.repeat(opt.nSample, 1)
            G_sample = netG(z, text_feat)
            gen_feat[idx*opt.nSample:(idx+1)*opt.nSample] = G_sample
            gen_label = np.hstack((gen_label, np.ones([opt.nSample])*alab))
            replay_z.append(z)

    replay_z = torch.cat(replay_z, dim=0)
    gen_label = torch.from_numpy(gen_label).long()
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
                res_arr = each_res + [0.0]*(num_item-len(each_res))
            else:
                res_arr = each_res
            log_list(res_arr, log)


def weights_init(m):
    classname = m.__class__.__name__
    if classname == "Linear":
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='leaky_relu')
    # if classname.find('Conv') != -1:
    # elif classname.find('BatchNorm') != -1:
    #     nn.init.normal_(m.weight.data, 1.0, 0.02)
    #     nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    for _ in range(opt.repeat):
        train()
