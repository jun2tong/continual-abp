import pdb
import h5py
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import torch


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i

    return mapped_label


class DATA_LOADER(object):
    def __init__(self, opt):

        if opt.dataset == 'imageNet1K':
            self.read_matimagenet(opt)
        elif opt.dataset.lower() == "mnist":
            self.read_pt(opt)
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
