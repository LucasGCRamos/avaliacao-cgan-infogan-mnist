
"""
train/train_infogan.py
Training loop for convolutional InfoGAN.
"""
import torch
import torch.nn as nn
from torch import optim
from models.infogan_conv import InfoGenConv, InfoDiscQ

def train_infogan(train_loader, device, latent_dim=62, cat_dim=10, cont_dim=2, epochs=10, lr=2e-4):
    G = InfoGenConv(latent_dim=latent_dim, cat_dim=cat_dim, cont_dim=cont_dim).to(device)
    DQ = InfoDiscQ(cat_dim=cat_dim, cont_dim=cont_dim).to(device)
    optG = optim.Adam(G.parameters(), lr=lr, betas=(0.5,0.999))
    optD = optim.Adam(DQ.parameters(), lr=lr, betas=(0.5,0.999))
    bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    for ep in range(epochs):
        total_d, total_g, total_q = 0.0, 0.0, 0.0
        for x,_ in train_loader:
            x = x.to(device)
            b = x.size(0)
            real = torch.ones(b,1, device=device)
            fake = torch.zeros(b,1, device=device)

            # Sample latents
            z = torch.randn(b, latent_dim, device=device)
            cat_idx = torch.randint(0, cat_dim, (b,), device=device)
            c_cat = torch.zeros(b, cat_dim, device=device).scatter_(1, cat_idx.unsqueeze(1), 1.0)
            c_cont = torch.randn(b, cont_dim, device=device) * 0.1

            # Train D (real and fake)
            fake_imgs = G(z, c_cat, c_cont)
            d_real, _ = DQ(x)
            d_fake, q_fake = DQ(fake_imgs.detach())
            d_loss = bce(d_real, real) + bce(d_fake, fake)

            optD.zero_grad()
            # Q-loss (predict categorical)
            q_loss = ce(q_fake[:, :cat_dim], cat_idx)
            (d_loss + q_loss).backward()
            optD.step()

            # Train G
            d_fake, q_fake = DQ(fake_imgs)
            g_loss = bce(d_fake, real)
            q_loss_g = ce(q_fake[:, :cat_dim], cat_idx) + mse(q_fake[:, cat_dim:], c_cont)
            optG.zero_grad()
            (g_loss + q_loss_g).backward()
            optG.step()

            total_d += d_loss.item(); total_g += g_loss.item(); total_q += q_loss.item()

        print(f"[InfoGAN] Epoch {ep+1}/{epochs} D_loss={total_d/len(train_loader):.4f} G_loss={total_g/len(train_loader):.4f} Q_loss={total_q/len(train_loader):.4f}")

    return G
