import argparse
import os
import glob
import numpy as np
import time
import torch
import torch.nn as nn

import albumentations as transforms
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet50

from pipeline.imagenet_dataset import TrainDataset
from pipeline.util import AverageMeter, timeSince

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def init_logger(log_file="./train1.log"):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, gradient_acc_step=1):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    # gradient_acc_step = 2
    optimizer.zero_grad()
    for step, (x_mb, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        img = x_mb.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        img, targets_a, targets_b, lam = mixup_data(img, labels, 1.0, device)
        y_preds = model(img)
        loss = mixup_criterion(criterion["cls"], y_preds, targets_a, targets_b, lam)

        # record loss
        losses.update(loss.item(), batch_size)

        if gradient_acc_step > 1:
            loss = loss / gradient_acc_step

        loss.backward()
        if (step+1) % gradient_acc_step == 0:
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
        if step % 200 == 0 or step == (len(train_loader) - 1):
            print_str = (
                f"Epoch: [{epoch+1}][{step}/{len(train_loader)}] "
                f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                f"Elapsed {timeSince(start, float(step+1)/len(train_loader)):s} "
                f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
                f"lr: {scheduler.get_last_lr()[0]:.7f}"
            )
            print(print_str)
    scheduler.step()
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to evaluation mode
    model.eval()
    corr = []
    start = end = time.time()
    for step, (x_mb, labels) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        img = x_mb.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            y_preds = model(img)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)

        # record accuracy
        preds = torch.argmax(torch.softmax(y_preds, dim=1), dim=1).to("cpu").numpy()
        correct = np.equal(preds, labels.to("cpu").numpy())
        corr.append(correct)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % 200 == 0 or step == (len(valid_loader) - 1):
            print_str = (
                f"EVAL: [{step}/{len(valid_loader)}] "
                f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                f"Elapsed {timeSince(start, float(step+1)/len(valid_loader)):s} "
                f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
            )
            print(print_str)

    num_corr = np.concatenate(corr)
    acc = num_corr.sum() / float(len(num_corr))
    return losses.avg, acc


def train_loop(train_ds, valid_ds):

    LOGGER.info(f"========== Training Starting ==========")

    # ====================================================
    # loader
    # ====================================================

    train_loader = DataLoader(
        train_ds,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # ====================================================
    # model & optimizer
    # ====================================================
    model = resnet50()

    # checkpoint = torch.load("best_706.pth")
    # model.load_state_dict(checkpoint["model"])
    # teacher_model = resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 1000)
    
    # teacher_model.eval()
    # for param in teacher_model.parameters():
    #     param.requires_grad = False

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    # teacher_model.to(device)

    if torch.cuda.device_count() > 1:
        optimizer = torch.optim.Adam(model.module.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    # optimizer.load_state_dict(checkpoint["optimizer"])
    # ====================================================
    # scheduler
    # ====================================================
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=CFG.min_lr)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.lr, epochs=CFG.epochs, 
    #                                                 steps_per_epoch=len(train_loader), 
    #                                                 final_div_factor=50,
    #                                                 cycle_momentum=False)
    # ====================================================
    # loop
    # ====================================================
    criterion = {"cls": nn.CrossEntropyLoss(), "dist": nn.BCEWithLogitsLoss()}
    # criterion = nn.CrossEntropyLoss()

    best_score = 0.0
    best_loss = np.inf
    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, 1)

        # eval
        avg_val_loss, acc = valid_fn(valid_loader, model, criterion["cls"], device)

        elapsed = time.time() - start_time

        LOGGER.info(f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s")
        LOGGER.info(f"Epoch {epoch+1} - Score: {acc:.4f}")

        if avg_val_loss < best_loss:
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
            if acc > best_score:
                best_score = acc
            LOGGER.info(f"Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Best Score {best_score:.4f} Model")
            if torch.cuda.device_count() > 1:
                torch.save({"model": model.module.state_dict(), "optimizer": optimizer.state_dict()}, f"best_res50.pth")
            else:
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, f"best_res50.pth")


def mixup_data(x, y, alpha=1.0, device="cpu"):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="continual-abp")
    parser.add_argument("--data", default="~/project/data/ILSVR2012/",
                        type=str, help="datapath")
    args = parser.parse_args()
    DATADIR = args.data
    # DATADIR = os.path.expanduser("~/project/data/ILSVR2012/")
    train_data_path = os.path.join(DATADIR,"train")
    val_data_path = os.path.join(DATADIR,"val")

    # Select the first 500
    class_lst = glob.glob(os.path.join(train_data_path,'*'))[:500]
    label_map = {key.split("/")[-1]: idx for idx, key in enumerate(class_lst)}
    
    train_img_path = [glob.glob(f"{apth}/*.JPEG") for apth in class_lst]
    train_img_path = [img for apth in train_img_path for img in apth]

    valid_img_path = []

    for each_key in label_map.keys():
        val_path = os.path.join(val_data_path, each_key)
        valid_img_path.append(glob.glob(f"{val_path}/*.JPEG"))
    valid_img_path = [img for apth in valid_img_path for img in apth]

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0, max_pixel_value=255.0,
    )

    augmentation = [
            transforms.RandomResizedCrop(256, 256, scale=(0.9, 1.0), p=1),
            transforms.HorizontalFlip(p=0.5),
            # transforms.ColorJitter(),
            transforms.RandomCrop(224, 224, True, p=1.),
            normalize,
            ToTensorV2(),
        ]
    train_transform = transforms.Compose(augmentation, p=1.0)

    valid_transform = transforms.Compose([transforms.Resize(224, 224), normalize, ToTensorV2()], p=1.0)

    train_ds = TrainDataset(train_img_path, label_map, transform=train_transform)
    valid_ds = TrainDataset(valid_img_path, label_map, transform=valid_transform)

    class CFG:
        lr = 0.0005
        wd = 1e-5
        epochs = 50
        min_lr=0.000005


    LOGGER = init_logger(f"pre-train-imagenet.log")

    train_loop(train_ds, valid_ds)
