import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torch.nn.init as init
from sklearn.preprocessing import MinMaxScaler 
import sys
import pdb


def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias, 0.0)


class CLASSIFIER:
    # train_Y is interger 
    def __init__(self, _train_X, _train_Y, data_loader, _nclass, _cuda, _lr=0.001, _beta1=0.5, _nepoch=20,
                 _batch_size=100, MCA=True, device="cpu"):
        self.train_X = _train_X
        self.train_Y = _train_Y 
        self.test_seen_feature = data_loader.test_seen_feature
        self.test_seen_label = data_loader.test_seen_label 
        self.seenclasses = data_loader.seenclasses
        self.ntrain_class = data_loader.ntrain_class
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _train_X.shape[1]
        self.cuda = _cuda
        self.MCA = MCA
        self.model = LINEAR_LOGSOFTMAX(self.input_dim, self.nclass)
        self.model.apply(weights_init)
        self.criterion = nn.NLLLoss()
        self.device = device
        self.input = torch.FloatTensor(_batch_size, self.input_dim) 
        self.label = torch.LongTensor(_batch_size) 
        
        self.lr = _lr
        self.beta1 = _beta1
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))

        if self.cuda:
            self.model.to(device)
            self.criterion.to(device)
            self.input = self.input.to(device)
            self.label = self.label.to(device)

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.shape[0]

        self.acc = self.fit()

    def fit(self):
        best_acc = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                # self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                batch_input = batch_input.to(self.device)
                batch_label = batch_label.to(self.device)
                output = self.model(batch_input)
                loss = self.criterion(output, batch_label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            acc = self.val(self.test_seen_feature, self.test_seen_label)

            if acc > best_acc:
                best_acc = acc
        return best_acc * 100

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            #print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0), torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            #print(start, end)
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]

    def val_gzsl(self, test_X, test_label):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        with torch.no_grad():
            for i in range(0, ntest, self.batch_size):
                end = min(ntest, start+self.batch_size)
                if self.cuda:
                    output = self.model(test_X[start:end].cuda())
                else:
                    output = self.model(test_X[start:end])
                _, predicted_label[start:end] = torch.max(output.data, 1)
                start = end
        if self.MCA:
            acc = self.eval_MCA(predicted_label.numpy(), test_label.numpy())
        else:
            acc = (predicted_label.numpy() == test_label.numpy()).mean()
        return acc

    def eval_MCA(self, preds, y):
        cls_label = np.unique(y)
        acc = list()
        for i in cls_label:
            acc.append((preds[y == i] == i).mean())
        return np.asarray(acc).mean()

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class += torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx)
        acc_per_class /= target_classes.size(0)
        return acc_per_class 

    # test_label is integer 
    def val(self, test_X, test_label):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        with torch.no_grad():
            for i in range(0, ntest, self.batch_size):
                end = min(ntest, start+self.batch_size)
                if self.cuda:
                    output = self.model(test_X[start:end].cuda())
                else:
                    output = self.model(test_X[start:end])
                _, predicted_label[start:end] = torch.max(output.data, 1)
                start = end
            acc = (predicted_label.numpy() == test_label.numpy()).mean()
        return acc

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx)
        return acc_per_class.mean()


class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, 512),
                                nn.ReLU(),
                                nn.Linear(512, nclass))
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x): 
        o = self.logic(self.fc(x))
        return o  
