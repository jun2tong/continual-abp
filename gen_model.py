import torch
import torch.nn as nn


class CifarNet(nn.Module):
    def __init__(self, latent_dim, nc, num_k):
        super(CifarNet, self).__init__()
        self.num_nodes = latent_dim + num_k
        self.decoder = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=self.num_nodes, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=128, out_channels=nc, kernel_size=4, stride=2, padding=1))
        # output of main module --> Image (Cx32x32)

        self.output = nn.Sigmoid()

    def forward(self, z, y=None):
        in_vec = torch.cat([z, y], dim=1)
        out = self.decoder(in_vec.view(-1, self.num_nodes, 1, 1))
        return out


class FeaturesGenerator(torch.nn.Module):
    def __init__(self, latent_dim, num_k, out_dim, hidden_dim=1024, num_c=0):
        self.num_nodes = latent_dim + num_k + num_c
        super(FeaturesGenerator, self).__init__()
        self.main = nn.Sequential(nn.Linear(self.num_nodes, hidden_dim),
                                  nn.LeakyReLU(0.2, True),
                                #   nn.ReLU(True),
                                  nn.Linear(hidden_dim, hidden_dim),
                                  nn.LeakyReLU(0.2, True),
                                  nn.Linear(hidden_dim, out_dim),
                                  nn.Sigmoid()
                                  )
        
    def forward(self, x, y, cls_c=None):
        if cls_c is not None:
            in_vec = torch.cat([x, y, cls_c], dim=1)
        else:
            in_vec = torch.cat([x, y], dim=1)
        out = self.main(in_vec)
        return out


class LinearCLS(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LinearCLS, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o


class InferenceQYZ(nn.Module):
    def __init__(self, latent_dim, num_k):
        super(InferenceQYZ, self).__init__()
        self.infer_cat = nn.Linear(latent_dim, num_k)

    def forward(self, x):
        logits = self.infer_cat(x)
        prob_cat = torch.softmax(logits, dim=1)
        return logits, prob_cat


# our generator
class ConditionalGenerator(nn.Module):
    def __init__(self, opt):
        super(ConditionalGenerator, self).__init__()
        # self.main = nn.Sequential(nn.Linear(opt.Z_dim, opt.gh_dim),
        #                           nn.LeakyReLU(0.2, True),
        #                           nn.Linear(opt.gh_dim, opt.X_dim),
        #                           nn.Sigmoid())
        self.main = nn.Sequential(nn.Linear(opt.Z_dim, 400),
                                  nn.ReLU(True),
                                  nn.Linear(400, 400),
                                  nn.ReLU(True),
                                  nn.Linear(400, 400),
                                  nn.ReLU(True),
                                  nn.Linear(400, opt.X_dim),
                                  nn.Sigmoid())

    def forward(self, z, c):
        # in_vec = torch.cat([z, c], dim=1)
        output = self.main(z)
        return output
