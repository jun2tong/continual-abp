import os
import glob
import pickle
import h5py
import numpy as np
import scipy.io as sio
import torch

from sklearn import preprocessing
from PIL import Image
from torch.utils import data


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i

    return mapped_label


def read_pickle(data_path, file_list):
    data = []
    targets = []
    # now load the picked numpy arrays
    for file_name, checksum in file_list:
        file_path = f"{data_path}/{file_name}"
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            data.append(entry['data'])
            if 'labels' in entry:
                targets.extend(entry['labels'])
            else:
                targets.extend(entry['fine_labels'])
    data = np.vstack(data).reshape(-1, 3, 32, 32)
    data = data.transpose((0, 2, 3, 1))  # convert to HWC
    return data, targets


class DATA_LOADER(object):
    def __init__(self, opt):

        # if opt.dataset == 'imageNet1K':
        #     self.read_matimagenet(opt)
        # if opt.dataset.lower() == "mnist":
        #     self.read_pt(opt)
        if opt.pca > 0:
            self.read_pca(opt, opt.dataset)
        elif opt.dataset.lower() == 'cifar10feas':
            feas_name = f"cifar-10-batches-py/{opt.image_embedding}"
            self.read_cifar10_feas(opt, feas_name)
        elif opt.dataset.lower() == 'cifar100feas':
            feas_name = f"cifar-100-python/{opt.image_embedding}"
            self.read_cifar10_feas(opt, feas_name)
        elif opt.dataset.lower() == "cub":
            self.read_train_test(opt, opt.dataset)
        else:
            self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.feature_dim = self.train_feature.shape[1]
        self.att_dim = self.attribute.shape[1]
        self.text_dim = self.att_dim

    def read_cifar10_feas(self, opt, name):
        # name is of the form: cifar-10-batches-py/resnet56_feas.tar
        checkpoint = torch.load(f"{opt.dataroot}/{name}.tar")
        x_train = checkpoint["x_train"].numpy()
        x_valid = checkpoint["x_valid"].numpy()
        self.train_label = checkpoint['y_train']  # long tensor
        self.test_seen_label = checkpoint["y_valid"]  # long tensor

        scaler = preprocessing.MinMaxScaler()
        _train_feature = scaler.fit_transform(x_train)
        _test_seen_feature = scaler.transform(x_valid)
        self.train_feature = torch.from_numpy(_train_feature)
        mx = self.train_feature.max()
        self.train_feature.mul_(1/mx)
        self.test_seen_feature = torch.from_numpy(_test_seen_feature)
        self.test_seen_feature.mul_(1/mx)
        self.ntrain_class = torch.unique(self.test_seen_label).size(0)
        self.attribute = torch.randn(10, 10).numpy()

    def read_pca(self, opt, name):
        checkpoint = torch.load(f"{opt.dataroot}/cifar-100-python/pca{opt.pca}.tar")
        _train_feature = checkpoint["train_x"]
        _test_seen_feature = checkpoint['test_x']
        self.train_label = checkpoint['train_y']  # long tensor
        self.test_seen_label = checkpoint["test_y"]  # long tensor

        scaler = preprocessing.MinMaxScaler()
        _train_feature = scaler.fit_transform(_train_feature)
        _test_seen_feature = scaler.transform(_test_seen_feature)
        self.train_feature = torch.from_numpy(_train_feature).float()
        mx = self.train_feature.max()
        self.train_feature.mul_(1 / mx)
        self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
        self.test_seen_feature.mul_(1 / mx)
        self.ntrain_class = torch.unique(self.test_seen_label).size(0)
        self.attribute = torch.randn(10, 10).numpy()
        # matcontent = sio.loadmat(f"{opt.dataroot}/{opt.dataset}/att_splits.mat")
        # self.attribute = torch.from_numpy(matcontent['att'].T).float()

    def read_train_test(self, opt, name):
        checkpoint = torch.load(f"{opt.dataroot}/{name}/train-test-split.tar")
        _train_feature = checkpoint["train_x"].numpy()
        _test_seen_feature = checkpoint['test_x'].numpy()
        self.train_label = checkpoint['train_y']  # long tensor
        self.test_seen_label = checkpoint["test_y"]  # long tensor

        scaler = preprocessing.MinMaxScaler()
        _train_feature = scaler.fit_transform(_train_feature)
        _test_seen_feature = scaler.transform(_test_seen_feature)
        self.train_feature = torch.from_numpy(_train_feature).float()
        mx = self.train_feature.max()
        self.train_feature.mul_(1 / mx)
        self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
        self.test_seen_feature.mul_(1 / mx)
        self.ntrain_class = torch.unique(self.test_seen_label).size(0)

        matcontent = sio.loadmat(f"{opt.dataroot}/{opt.dataset}/att_splits.mat")
        self.attribute = torch.from_numpy(matcontent['att'].T).float()

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(f"{opt.dataroot}/{opt.dataset}/{opt.image_embedding}.mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(f"{opt.dataroot}/{opt.dataset}/att_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1

        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        self.test_unseen_loc = test_unseen_loc
        self.attribute = torch.from_numpy(matcontent['att'].T).float()

        scaler = preprocessing.MinMaxScaler()
        _train_feature = scaler.fit_transform(feature[trainval_loc])
        _test_seen_feature = scaler.transform(feature[test_seen_loc])
        _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
        self.train_feature = torch.from_numpy(_train_feature).float()
        mx = self.train_feature.max()
        self.train_feature.mul_(1 / mx)
        self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
        self.test_unseen_feature.mul_(1 / mx)
        self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
        self.test_seen_feature.mul_(1 / mx)

        self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
        self.train_label = torch.from_numpy(label[trainval_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        # self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class).long()

        # self.train_label = map_label(self.train_label, self.seenclasses)
        # self.test_seen_label = map_label(self.test_seen_label, self.seenclasses)
        # self.test_unseen_label = map_label(self.test_unseen_label, self.unseenclasses)

        self.train_att = self.attribute[self.seenclasses]
        self.test_att = self.attribute[self.unseenclasses]


class CORE50(object):
    def __init__(self, task_id, model='resnet18'):
        checkpoint = torch.load(f"data/core50/{model}/core50feas_task{task_id}.tar")
        x_train = checkpoint["x_train"].numpy()
        self.train_label = checkpoint['y_train'].long()
        checkpoint = torch.load(f"data/core50/{model}/core50feas_test_task{task_id}.tar")
        self.test_seen_label = checkpoint["y_valid"].long()
        x_valid = checkpoint["x_valid"].numpy()

        scaler = preprocessing.MinMaxScaler()
        _train_feature = scaler.fit_transform(x_train)
        _test_seen_feature = scaler.transform(x_valid)
        self.train_feature = torch.from_numpy(_train_feature)
        self.test_seen_feature = torch.from_numpy(_test_seen_feature)
        self.ntrain_class = 10
        self.attribute = torch.randn(10, 10).numpy()


class FeatDataLayer(object):
    def __init__(self, label, feat_data, latent_z, mini_batchsize):
        """Set the roidb to be used by this layer during training."""
        assert len(label) == feat_data.shape[0]
        self.mini_batchsize = mini_batchsize
        self.init_data = feat_data
        self.feat_data = feat_data
        self.init_label = label
        self.label = label
        self.latent_z = latent_z
        self.num_obs = len(label)
        self.perm = None
        self.cur = 0

        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self.perm = np.random.permutation(np.arange(self.num_obs))
        self.cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self.cur + self.mini_batchsize >= len(self.label):
            self._shuffle_roidb_inds()
            # self.epoch += 1

        db_inds = self.perm[self.cur:self.cur + self.mini_batchsize]
        self.cur += self.mini_batchsize

        return db_inds

    def add_data(self, new_labels, new_feas):
        self.feat_data = np.concatenate([self.init_data, new_feas],axis=0)
        self.label = np.concatenate([self.init_label, new_labels])
        self.num_obs = len(self.label)
        self._shuffle_roidb_inds()

    def forward(self):
        if self.cur + self.mini_batchsize >= self.num_obs:
            db_inds = self.perm[self.cur:]
            self._shuffle_roidb_inds()
        else:
            db_inds = self.perm[self.cur:self.cur + self.mini_batchsize]
            self.cur += self.mini_batchsize

        minibatch_feat = self.feat_data[db_inds]
        minibatch_label = self.label[db_inds]
        minibatch_z = self.latent_z[db_inds]
        blobs = {'data': minibatch_feat, 'labels': minibatch_label, 
                 'idx': db_inds, 'latent_z': minibatch_z}
        return blobs

    def get_whole_data(self):
        blobs = {'data': self.feat_data, 'labels': self.label}
        return blobs


class StreamDataLayer(object):
    def __init__(self, label, feat_data,  mini_batchsize):
        """Set the roidb to be used by this layer during training."""
        assert len(label) == feat_data.shape[0]

        sort_res = torch.sort(label)

        self.mini_batchsize = mini_batchsize
        self.feat_data = feat_data[sort_res[1]]
        self.label = label[sort_res[1]]
        self.num_obs = len(label)
        self.cur = 0
        self.iter = 0

    def has_next_batch(self):
        if self.cur < self.num_obs:
            return True
        else:
            return False

    def get_iteration(self):
        return self.iter

    def get_cur_idx(self):
        return self.cur

    def get_batch_data(self):
        if self.cur + self.mini_batchsize >= self.num_obs:
            db_inds = np.arange(self.cur, self.num_obs)
            self.cur = self.num_obs
        else:
            db_inds = np.arange(self.cur, self.cur+self.mini_batchsize)
            self.cur += self.mini_batchsize

        minibatch_feat = self.feat_data[db_inds]
        minibatch_label = self.label[db_inds]
        self.iter += 1
        blobs = {'data': minibatch_feat, 'labels': minibatch_label, 'idx': db_inds}
        return blobs

    def get_whole_data(self):
        blobs = {'data': self.feat_data, 'labels': self.label}
        return blobs
