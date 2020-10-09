import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.utils as tutils
from torch.utils.data import DataLoader

import torchvision.transforms as tv_transforms

from tiny_resnet import resnet32
# from naiveresnet import NaiveResNet
# from resnet import resnet32
from dataset_GBU import TinyImageNet

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

normalize = tv_transforms.Normalize(mean=(.485, .456, .406), std=(.229, .224, .225))

aug1 = tv_transforms.RandomApply([tv_transforms.ColorJitter(0.2),
                                  tv_transforms.RandomRotation(15)], p=0.5)

aug2 = tv_transforms.RandomApply([tv_transforms.RandomHorizontalFlip(),
                                  tv_transforms.RandomRotation(15),
                                  tv_transforms.RandomResizedCrop(64)], p=0.5)

augmentation = tv_transforms.RandomChoice([
    aug1,
    aug2,
    tv_transforms.RandomVerticalFlip(),
    tv_transforms.RandomHorizontalFlip(),
    tv_transforms.RandomRotation(15),
    tv_transforms.RandomResizedCrop(64)])

img_transform = tv_transforms.Compose([tv_transforms.ToPILImage(),
                                       augmentation,
                                       tv_transforms.ToTensor(),
                                       normalize])
val_transform = tv_transforms.Compose([tv_transforms.ToTensor(),
                                       normalize])

train_dataset = TinyImageNet("data/tiny-imagenet-200", split='train', in_memory=True, transform=img_transform)
val_dataset = TinyImageNet("data/tiny-imagenet-200", split='val', in_memory=True, transform=val_transform)

resnet = resnet32()
# resnet = NaiveResNet(200)
# checkpoint = torch.load("pre-train_res_tiny-imagenet.tar")
# resnet.load_state_dict(checkpoint["resnet_state_dict"])
resnet = resnet.to(device)

# optimizer = torch.optim.Adam(resnet.parameters(), lr=0.01, weight_decay=0.0001)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [150, 350, 500], gamma=0.1)
optimizer = torch.optim.Adam(resnet.parameters(), lr=0.1, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 250, 600], gamma=0.1)
ce_loss = nn.CrossEntropyLoss()

trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
validloader = DataLoader(val_dataset, batch_size=128, num_workers=4)

max_epochs = 1000
for epoch in range(max_epochs):

    epoch_loss = 0.0
    print_eval = False
    if epoch % 50 == 0:
        print_eval = True
    resnet.train()
    for idx, (data, target) in enumerate(trainloader):
        data, target = data.float().to(device), target.long().to(device)
        optimizer.zero_grad()
        output = resnet(data)
        batch_loss = ce_loss(output, target)
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.detach()

        if idx % 400 == 0 and print_eval:
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

            valid_acc = num_cor.float()/len(val_dataset) * 100
            print(f"Eval -- epoch {epoch}; loss: {epoch_loss: .4f}; valid_acc: {valid_acc: .4f}%")
        torch.save({"resnet_state_dict": resnet.state_dict()}, "pre-train_res_tinyimagenet200.tar")
torch.save({"resnet_state_dict": resnet.state_dict()}, "pre-train_res_tinyimagenet200.tar")
