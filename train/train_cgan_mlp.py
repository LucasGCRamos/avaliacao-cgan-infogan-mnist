import torch
import torch.nn as nn
import torch.optim as optim

from models.cgan_mlp import Generator, Discriminator

def train_cgan_mlp(train_loader, device, latent_dim=100, epochs=5, lr=2e-4):
    G = Generator(z_dim=latent_dim).to(device)
    D = Discriminator().to(device)

    optG = optim.Adam(G.parameters(), lr=lr)
    optD = optim.Adam(D.parameters(), lr=lr)

    loss_fn = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        total_d, total_g = 0.0, 0.0

        for x, y in train_loader:
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)
            b = x.size(0)

            real = torch.ones(b, 1, device=device)
            fake = torch.zeros(b, 1, device=device)

            # --------------------
            # Train Discriminator
            # --------------------
            z = torch.randn(b, latent_dim, device=device)
            fake_imgs = G(z, y)

            d_real = D(x, y)
            d_fake = D(fake_imgs.detach(), y)

            d_loss = (
                loss_fn(d_real, real) +
                loss_fn(d_fake, fake)
            )

            optD.zero_grad()
            d_loss.backward()
            optD.step()

            # --------------------
            # Train Generator
            # --------------------
            z = torch.randn(b, latent_dim, device=device)
            fake_imgs = G(z, y)
            g_loss = loss_fn(D(fake_imgs, y), real)

            optG.zero_grad()
            g_loss.backward()
            optG.step()

            total_d += d_loss.item()
            total_g += g_loss.item()

        print(f"[cGAN MLP] Epoch {ep+1}/{epochs} D={total_d/len(train_loader):.4f} G={total_g/len(train_loader):.4f}")

    return G
