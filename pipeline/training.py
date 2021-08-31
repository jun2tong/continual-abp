import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models.gen_model import FeaturesGenerator
from models.cifar_resnet import resnet32

device = "cuda:0"


def train_fe(train_loader, test_loader, task_id, prev_loc, opt):
    model = resnet32(50)
    if task_id > 0:
        model.load_state_dict(torch.load(prev_loc)["model_state_dict"])
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.nepoch_fe, eta_min=0.0001)
    criterion = nn.CrossEntropyLoss()

    track_train_loss = []
    for epc in range(opt.nepoch_fe):
        model.train()
        all_loss = 0
        for iter_idx, (mb_idx, x_mb, y_mb) in enumerate(train_loader):
            x_mb = x_mb.to(device)
            y_mb = y_mb.to(device)
            logits = model(x_mb)
            
            ce_loss = criterion(logits, y_mb)
            optimizer.zero_grad()
            ce_loss.backward()
            optimizer.step()

            all_loss += ce_loss.detach()
            track_train_loss.append(ce_loss.detach().item())
            
        all_loss /= float(iter_idx+1)
        scheduler.step()
        if (epc+1) % 20 == 0:
            with torch.no_grad():
                acc = eval_model_fe(model, test_loader)
            print(f"Epoch {epc+1} cls loss:{all_loss.item():.4f} test acc:{acc:.4f}")
    
    tar_name = "cifar100-pretrained.tar"
    torch.save({"model_state_dict": model.state_dict()}, tar_name)
    return tar_name


def train_generator(netG, train_z, dataloader, opt_g, latent_dim, num_k):
    total_niter = int(dataloader.num_obs/128) * 50
    for it in range(total_niter+1):
        blobs = dataloader.forward()
        feat_data = blobs['data']  # image data
        y_mb = torch.from_numpy(blobs['labels'].astype(int))  # class labels
        idx = blobs['idx'].astype(int)
        x_mb = torch.from_numpy(feat_data).float().to(device)
        z_mb = train_z[idx]
        z_mb.requires_grad_()
        prev_z = torch.from_numpy(blobs["latent_z"]).to(device)

        optimizer_z = torch.optim.Adam([z_mb], lr=0.0002, weight_decay=0., betas=(0.5, 0.999))

        # Alternate update weights w and infer latent_batch z
        batch_loss = 0
        for em_step in range(1):  # EM_STEP
            opt_g.zero_grad()
            one_hot_y = torch.eye(num_k)[y_mb]
            recon_x = netG(z_mb, one_hot_y.to(device))
            recon_loss = get_recon_loss(recon_x, x_mb, 0.1)  # Reconstruction Loss
            prior_loss = get_prior_loss(z_mb, prev_z)

            gloss = recon_loss + prior_loss
            gloss /= x_mb.size(0)
            gloss.backward()
            opt_g.step()
            srmc_loss = 0
            for ls_step in range(20):
                optimizer_z.zero_grad()
                u_tau = torch.randn(z_mb.size(0), latent_dim).float().to(device)

                one_hot_y = torch.eye(num_k)[y_mb]
                recon_x = netG(z_mb, one_hot_y.to(device))
                recon_loss = get_recon_loss(recon_x, x_mb, 0.1)
                prior_loss = get_prior_loss(z_mb, prev_z)

                loss = recon_loss + prior_loss
                loss /= x_mb.size(0)
                loss = loss * (0.3*0.3)/2
                loss.backward()
                optimizer_z.step()
                # TODO: Maybe implement metropolis hastings selection here?
                if ls_step < 1 and it < int(dataloader.num_obs/128):
                    z_mb.data += u_tau * 0.3
                srmc_loss += loss.detach()

            train_z[idx,] = z_mb.data
            batch_loss += (srmc_loss / 20) + gloss.detach()

        if it % 500 == 0 and it:
            log_text = f'Iter-[{it}/{total_niter}]; loss: {batch_loss :.4f}'
            print(log_text)

    return netG, train_z


def eval_model_fe(model, dataloader):
    acc = 0
    num_corr = 0
    count = 0
    model.eval()
    for iter_idx, (_, x_mb, y_mb) in enumerate(dataloader):
        x_mb = x_mb.to(device)
        y_mb = y_mb.to(device)
        logits = model(x_mb)
        pred = torch.argmax(torch.softmax(logits, dim=1),dim=1)
        corr = torch.sum(torch.eq(pred, y_mb))
        num_corr += corr
        count += x_mb.size(0)
    acc = num_corr / float(count)
    return acc


def get_recon_loss(pred, x, sigma):
    recon_loss = nn.functional.mse_loss(pred, x, reduction='sum') / (2*sigma*sigma)
    return recon_loss


def get_prior_loss(z, prev_rep):
    log_pdf = nn.functional.mse_loss(z,prev_rep,reduction='sum') * 0.5
    return log_pdf
