
"""
models/cgan_conv.py
Convolutional Conditional GAN (cDCGAN).
Generator uses ConvTranspose2d. Discriminator uses Conv2d.
Conditioning is applied by concatenating a label-embedding as channels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CondGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, feature_maps=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # project label to a spatial map and concatenate to noise channels
        self.label_emb = nn.Embedding(num_classes, num_classes)

        # We'll transform (z + label_embed) -> feature map and upsample to 28x28
        self.fc = nn.Linear(latent_dim + num_classes, feature_maps*7*7)

        self.net = nn.Sequential(
            # input: (feature_maps) x 7 x 7
            nn.ConvTranspose2d(feature_maps, feature_maps//2, kernel_size=4, stride=2, padding=1), # 14x14
            nn.BatchNorm2d(feature_maps//2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps//2, feature_maps//4, kernel_size=4, stride=2, padding=1), # 28x28
            nn.BatchNorm2d(feature_maps//4),
            nn.ReLU(True),

            nn.Conv2d(feature_maps//4, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # labels: (B,)
        c = self.label_emb(labels)            # (B, num_classes)
        x = torch.cat([z, c], dim=1)          # (B, latent+num_classes)
        x = self.fc(x)                        # (B, feature_maps*7*7)
        x = x.view(x.size(0), -1, 7, 7)       # (B, feature_maps,7,7)
        img = self.net(x)
        return img


class CondDiscriminator(nn.Module):
    def __init__(self, num_classes=10, feature_maps=64):
        super().__init__()
        # project label to spatial map and concatenate as channels
        self.label_emb = nn.Embedding(num_classes, num_classes)

        # input channels = 1 (image) + num_classes (label map)
        in_ch = 1 + num_classes

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, feature_maps, kernel_size=4, stride=2, padding=1), #14x14
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps, feature_maps*2, kernel_size=4, stride=2, padding=1), #7x7
            nn.BatchNorm2d(feature_maps*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear((feature_maps*2)*7*7, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # create label maps: expand embedding to spatial dimensions and concat
        batch = img.size(0)
        c = self.label_emb(labels)                         # (B, num_classes)
        c_map = c.unsqueeze(2).unsqueeze(3).expand(-1, -1, img.size(2), img.size(3))  # (B, num_classes, H, W)
        x = torch.cat([img, c_map], dim=1)
        return self.net(x)
