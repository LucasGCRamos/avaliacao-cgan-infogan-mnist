
"""
train/train_cgan.py
Training loop for convolutional conditional GAN.
"""
import torch
import torch.nn as nn
from torch import optim
from models.cgan_conv import CondGenerator, CondDiscriminator

def train_cgan(train_loader, device, latent_dim=100, epochs=10, lr=2e-4):
    G = CondGenerator(latent_dim=latent_dim).to(device)
    D = CondDiscriminator().to(device)
    optG = optim.Adam(G.parameters(), lr=lr, betas=(0.5,0.999))
    optD = optim.Adam(D.parameters(), lr=lr, betas=(0.5,0.999))
    loss_fn = nn.BCELoss()

    for ep in range(epochs):
        total_d, total_g = 0.0, 0.0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            b = x.size(0)
            real = torch.ones(b,1, device=device)
            fake = torch.zeros(b,1, device=device)

            # Train D
            z = torch.randn(b, latent_dim, device=device)
            fake_imgs = G(z, y)
            d_loss = loss_fn(D(x,y), real) + loss_fn(D(fake_imgs.detach(), y), fake)
            optD.zero_grad(); d_loss.backward(); optD.step()

            # Train G
            z = torch.randn(b, latent_dim, device=device)
            fake_imgs = G(z, y)
            g_loss = loss_fn(D(fake_imgs, y), real)
            optG.zero_grad(); g_loss.backward(); optG.step()

            total_d += d_loss.item(); total_g += g_loss.item()

        print(f"[cCGAN] Epoch {ep+1}/{epochs} D_loss={total_d/len(train_loader):.4f} G_loss={total_g/len(train_loader):.4f}")

    return G
