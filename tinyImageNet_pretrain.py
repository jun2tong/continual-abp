import torch
import torch.nn as nn
import torchvision.datasets as Datasets
import torch.utils.data as tdata

from torch.utils.data import DataLoader

import torchvision.transforms as tv_transforms

# from resnet import resnet32


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")


class CustomTensorDataset(tdata.Dataset):
    def __init__(self, x_tensor, y_tensor, transform=None):
        self.tensors = (x_tensor, y_tensor)
        self.transform = transform
        self.length = x_tensor.shape[0]

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.length


img_transform = tv_transforms.Compose([tv_transforms.ToTensor()])


# Get first task data
print("##### getting first task data #####")
dataset = Datasets.CIFAR100(root="./data", train=True,
                            transform=img_transform, download=False)
dataloader = DataLoader(dataset, batch_size=128, num_workers=4, shuffle=True)

train_data = []
train_label = []
for x_mb, y_mb in dataloader:
    mask = y_mb < 10
    train_data.append(x_mb[mask])
    train_label.append(y_mb[mask])

train_data = torch.cat(train_data, dim=0)
train_label = torch.cat(train_label)
print("##### Done getting first task data #####")
# End get first task data


# Get first task validation data
print("##### getting task validation data #####")
dataset = Datasets.CIFAR100(root="./data", train=False,
                            transform=img_transform, download=False)
dataloader = DataLoader(dataset, batch_size=128, num_workers=4, shuffle=True)

valid_data = []
valid_label = []
for x_mb, y_mb in dataloader:
    mask = y_mb < 10
    valid_data.append(x_mb[mask])
    valid_label.append(y_mb[mask])

valid_data = torch.cat(valid_data, dim=0)
valid_label = torch.cat(valid_label)
print("##### Done getting task validation data #####")
# End get first task data

normalize = tv_transforms.Normalize(mean=(.485, .456, .406),
                                    std=(.229, .224, .225))

img_transform = tv_transforms.Compose([tv_transforms.ToPILImage(),
                                       tv_transforms.RandomHorizontalFlip(),
                                       tv_transforms.RandomCrop(32, 4),
                                       tv_transforms.ToTensor(), normalize])

val_transform = tv_transforms.Compose([tv_transforms.ToPILImage(),
                                       tv_transforms.ToTensor(), normalize])

train_ds = CustomTensorDataset(train_data, train_label, img_transform)
valid_ds = CustomTensorDataset(valid_data, valid_label, val_transform)

resnet = resnet32()
resnet = resnet.to(device)

optimizer = torch.optim.Adam(resnet.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 250], gamma=0.1)
# optimizer = torch.optim.Adam(resnet.parameters(), lr=0.01, weight_decay=0.0001)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 120], gamma=0.1)
ce_loss = nn.CrossEntropyLoss()

trainloader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
validloader = DataLoader(valid_ds, batch_size=128, num_workers=4)

max_epochs = 350
for epoch in range(max_epochs):

    epoch_loss = 0.0
    print_eval = False
    if epoch % 1 == 0:
        print_eval = True
    resnet.train()
    for idx, (x_mb, y_mb) in enumerate(trainloader):
        x_mb, y_mb = x_mb.float().to(device), y_mb.long().to(device)
        optimizer.zero_grad()
        output = resnet(x_mb)
        batch_loss = ce_loss(output, y_mb)
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.detach()

        if idx % 50 == 0 and print_eval:
            print(f'Train -- epoch: {epoch}, it: {idx}, loss: {batch_loss: .4f}')
    scheduler.step()
    epoch_loss /= float(idx+1)
    if print_eval:
        with torch.no_grad():
            resnet.eval()
            num_cor = 0
            for idx, (data, target) in enumerate(validloader):
                data, target = data.float().to(device), target.long().to(device)
                output = resnet(data)
                cls_pred = torch.argmax(output, dim=1)
                num_cor += torch.sum(torch.eq(cls_pred, target))

            valid_acc = num_cor.float()/len(valid_ds) * 100
            print(f"Eval -- epoch {epoch}; loss: {epoch_loss: .4f}; valid_acc: {valid_acc: .4f}%")
torch.save({"resnet_state_dict": resnet.state_dict()}, "res-cifar100-t1.tar")
