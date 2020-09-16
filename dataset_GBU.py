import pdb
import h5py
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import torch
import pickle


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

        if opt.dataset == 'imageNet1K':
            self.read_matimagenet(opt)
        elif opt.dataset.lower() == "mnist":
            self.read_pt(opt)
        elif opt.dataset.lower() == 'cifar10':
            self.read_cifar10(opt)
        elif opt.dataset.lower() == 'cifar10feas':
            self.read_cifar10_feas(opt)
        elif opt.dataset.lower() == 'cifar10feas_prepool':
            self.read_cifar10feas_prepool(opt)
        else:
            self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.feature_dim = self.train_feature.shape[1]
        self.att_dim = self.attribute.shape[1]
        self.text_dim = self.att_dim
        # self.tr_cls_centroid = np.zeros([self.seenclasses.shape[0], self.feature_dim], np.float32)
        # for i in range(self.seenclasses.shape[0]):
        #     self.tr_cls_centroid[i] = np.mean(self.train_feature[self.train_label == i].numpy(), axis=0)

    def read_matimagenet(self, opt):
        if opt.preprocessing:
            print('MinMaxScaler...')
            scaler = preprocessing.MinMaxScaler()
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = scaler.fit_transform(np.array(matcontent['features']))
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = scaler.transform(np.array(matcontent['features_val']))
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()
            matcontent = h5py.File('/BS/xian/work/data/imageNet21K/extract_res/res101_1crop_2hops_t.mat', 'r')
            feature_unseen = scaler.transform(np.array(matcontent['features']))
            label_unseen = np.array(matcontent['labels']).astype(int).squeeze() - 1
            matcontent.close()
        else:
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = np.array(matcontent['features'])
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = np.array(matcontent['features_val'])
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".mat")
        self.attribute = torch.from_numpy(matcontent['w2v']).float()
        self.train_feature = torch.from_numpy(feature).float()
        self.train_label = torch.from_numpy(label).long()
        self.test_seen_feature = torch.from_numpy(feature_val).float()
        self.test_seen_label = torch.from_numpy(label_val).long()
        self.test_unseen_feature = torch.from_numpy(feature_unseen).float()
        self.test_unseen_label = torch.from_numpy(label_unseen).long()
        self.ntrain = self.train_feature.size()[0]
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.train_class = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)

    def read_pt(self, opt):
        feature, label = torch.load(opt.dataroot + "/" + opt.dataset + "/processed/training.pt")
        num_train = feature.size(0)
        feature = feature.float()
        test_feature, test_label = torch.load(opt.dataroot + "/" + opt.dataset + "/processed/test.pt")
        num_test = test_feature.size(0)
        test_feature = test_feature.float()
        self.attribute = torch.randn(10, 10).numpy()
        self.train_att = torch.randn(10, 10).numpy()
        scaler = preprocessing.MinMaxScaler()
        _train_feature = scaler.fit_transform(feature.view(num_train, -1).numpy())
        _test_feature = scaler.transform(test_feature.view(num_test, -1).numpy())
        self.train_feature = torch.from_numpy(_train_feature).float()
        self.test_seen_feature = torch.from_numpy(_test_feature).float()
        self.ntrain_class = 10
        self.train_label = label.long()
        self.test_seen_label = test_label.long()

    def read_cifar10(self, opt):
        train_list = [
            ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
            ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
            ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
            ['data_batch_4', '634d18415352ddfa80567beed471001a'],
            ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
        ]
        test_list = [
            ['test_batch', '40351d587109b95175f43aff81a1287e'],
        ]
        train_data, train_targets = read_pickle(f"{opt.dataroot}/cifar-10-batches-py/", train_list)
        test_data, test_targets = read_pickle(f"{opt.dataroot}/cifar-10-batches-py/", test_list)
        self.attribute = torch.randn(10, 10).numpy()
        self.train_att = torch.randn(10, 10).numpy()
        _train_feature = train_data.astype(np.float) / 255.
        _test_feature = test_data.astype(np.float) / 255.
        self.train_feature = torch.from_numpy(_train_feature).float().permute(0,3,1,2)
        self.test_seen_feature = torch.from_numpy(_test_feature).float().permute(0,3,1,2)
        self.ntrain_class = 10
        self.train_label = torch.tensor(train_targets).long()
        self.test_seen_label = torch.tensor(test_targets).long()

    def read_cifar10_feas(self, opt):
        checkpoint = torch.load(f"{opt.dataroot}/cifar-10-batches-py/resnet56_feas.tar")
        x_train = checkpoint["x_train"].numpy()
        x_valid = checkpoint["x_valid"].numpy()
        self.train_label = checkpoint['y_train']  # long tensor
        self.test_seen_label = checkpoint["y_valid"]  # long tensor

        scaler = preprocessing.MinMaxScaler()
        _train_feature = scaler.fit_transform(x_train)
        _test_seen_feature = scaler.transform(x_valid)
        self.train_feature = torch.from_numpy(_train_feature)
        self.test_seen_feature = torch.from_numpy(_test_seen_feature)
        self.ntrain_class = 10
        self.attribute = torch.randn(10, 10).numpy()

    def read_cifar10feas_prepool(self, opt):
        checkpoint = torch.load(f"{opt.dataroot}/cifar10_midfeas.tar")
        x_train = checkpoint["x_train"].numpy()
        x_valid = checkpoint["x_valid"].numpy()
        self.train_label = checkpoint['y_train']  # long tensor
        self.test_seen_label = checkpoint["y_valid"]  # long tensor

        scaler = preprocessing.MinMaxScaler()
        _train_feature = scaler.fit_transform(x_train)
        _test_seen_feature = scaler.transform(x_valid)
        self.train_feature = torch.from_numpy(_train_feature)
        self.test_seen_feature = torch.from_numpy(_test_seen_feature)
        self.ntrain_class = 10
        self.attribute = torch.randn(10, 10).numpy()

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        # train_loc = matcontent['train_loc'].squeeze() - 1
        # val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        self.attribute = torch.from_numpy(matcontent['att'].T).float()
        if not opt.validation:
            scaler = preprocessing.MinMaxScaler()
            _train_feature = scaler.fit_transform(feature[trainval_loc])
            _test_seen_feature = scaler.transform(feature[test_seen_loc])
            # _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
            self.train_feature = torch.from_numpy(_train_feature).float()
            mx = self.train_feature.max()
            self.train_feature.mul_(1 / mx)
            self.train_label = torch.from_numpy(label[trainval_loc]).long()
            # self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
            # self.test_unseen_feature.mul_(1 / mx)
            self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
            self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
            self.test_seen_feature.mul_(1 / mx)
            self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        # else:
        #     self.train_feature = torch.from_numpy(feature[train_loc]).float()
        #     self.train_label = torch.from_numpy(label[train_loc]).long()
        #     self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
        #     self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class).long()

        self.train_label = map_label(self.train_label, self.seenclasses)
        self.test_seen_label = map_label(self.test_seen_label, self.seenclasses)
        self.train_att = self.attribute[self.seenclasses].numpy()


class CORE50(object):
    def __init__(self, task_id):
        checkpoint = torch.load(f"data/core50/core50feas_task{task_id}.tar")
        x_train = checkpoint["x_train"].numpy()
        self.train_label = checkpoint['y_train'].long()
        checkpoint = torch.load(f"data/core50/core50feas_test_task{task_id}.tar")
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
    def __init__(self, label, feat_data,  mini_batchsize):
        """Set the roidb to be used by this layer during training."""
        assert len(label) == feat_data.shape[0]
        self.mini_batchsize = mini_batchsize
        self.feat_data = feat_data
        self.label = label
        self.epoch = 0
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
            self.epoch += 1

        db_inds = self.perm[self.cur:self.cur + self.mini_batchsize]
        self.cur += self.mini_batchsize

        return db_inds

    def forward(self):
        if self.cur + self.mini_batchsize >= self.num_obs:
            db_inds = self.perm[self.cur:]
            self._shuffle_roidb_inds()
        else:
            db_inds = self.perm[self.cur:self.cur + self.mini_batchsize]
            self.cur += self.mini_batchsize

            # minibatch_feat = np.array([self.feat_data[i] for i in db_inds])
            # minibatch_label = np.array([self.label[i] for i in db_inds])
        minibatch_feat = self.feat_data[db_inds]
        minibatch_label = self.label[db_inds]
        blobs = {'data': minibatch_feat, 'labels': minibatch_label, 'idx': db_inds}
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
