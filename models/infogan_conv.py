
"""
models/infogan_conv.py
Convolutional InfoGAN (simplified)
Generator conditioned on categorical code (one-hot) and continuous codes (optionally).
Discriminator outputs real/fake and Q-network predicts categorical code.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoGenConv(nn.Module):
    def __init__(self, latent_dim=62, cat_dim=10, cont_dim=2, feature_maps=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.cat_dim = cat_dim
        self.cont_dim = cont_dim
        input_dim = latent_dim + cat_dim + cont_dim
        self.fc = nn.Linear(input_dim, feature_maps*7*7)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(feature_maps, feature_maps//2, 4, 2, 1), #14x14
            nn.BatchNorm2d(feature_maps//2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps//2, feature_maps//4, 4, 2, 1), #28x28
            nn.BatchNorm2d(feature_maps//4),
            nn.ReLU(True),

            nn.Conv2d(feature_maps//4, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z, c_cat, c_cont):
        # c_cat: one-hot (B, cat_dim), c_cont: (B, cont_dim)
        x = torch.cat([z, c_cat, c_cont], dim=1)
        x = self.fc(x).view(x.size(0), -1, 7, 7)
        return self.net(x)


class InfoDiscQ(nn.Module):
    def __init__(self, cat_dim=10, cont_dim=2, feature_maps=64):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, feature_maps, 4, 2, 1), #14x14
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feature_maps, feature_maps*2, 4, 2, 1), #7x7
            nn.BatchNorm2d(feature_maps*2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.flatten = nn.Flatten()
        self.disc = nn.Sequential(
            nn.Linear((feature_maps*2)*7*7, 1),
            nn.Sigmoid()
        )
        # Q-network: predict categorical logits + continuous (regression)
        self.q_net = nn.Sequential(
            nn.Linear((feature_maps*2)*7*7, 128),
            nn.ReLU(True),
            nn.Linear(128, cat_dim + cont_dim)
        )

    def forward(self, x):
        f = self.feature_extractor(x)
        f_flat = self.flatten(f)
        validity = self.disc(f_flat)
        q_out = self.q_net(f_flat)
        # q_out: first cat_dim logits, then cont_dim continuous predictions
        return validity, q_out
