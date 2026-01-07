import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models.infogan_mlp import Generator, DiscriminatorQ

def train_infogan_mlp(
    train_loader,
    device,
    latent_dim=62,
    cat_dim=10,
    cont_dim=2,
    epochs=5,
    lr=2e-4
):
    G = Generator(z_dim=latent_dim, c_dim=cat_dim + cont_dim).to(device)
    DQ = DiscriminatorQ(c_dim=cat_dim + cont_dim).to(device)

    optG = optim.Adam(G.parameters(), lr=lr)
    optDQ = optim.Adam(DQ.parameters(), lr=lr)

    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    for ep in range(epochs):
        total_d, total_g, total_q = 0.0, 0.0, 0.0

        for x, _ in train_loader:
            x = x.view(x.size(0), -1).to(device)
            b = x.size(0)

            real = torch.ones(b, 1, device=device)
            fake = torch.zeros(b, 1, device=device)

            z = torch.randn(b, latent_dim, device=device)

            cat_idx = torch.randint(0, cat_dim, (b,), device=device)
            c_cat = F.one_hot(cat_idx, cat_dim).float()
            c_cont = torch.randn(b, cont_dim, device=device) * 0.1
            c = torch.cat([c_cat, c_cont], dim=1)

            # --------------------
            # Train Discriminator + Q
            # --------------------
            fake_imgs = G(z, c)
            d_real, _ = DQ(x)
            d_fake, q_fake = DQ(fake_imgs.detach())

            d_loss = bce(d_real, real) + bce(d_fake, fake)
            q_loss = ce(q_fake[:, :cat_dim], cat_idx)

            optDQ.zero_grad()
            (d_loss + q_loss).backward()
            optDQ.step()

            # --------------------
            # Train Generator
            # --------------------
            d_fake, q_fake = DQ(fake_imgs)
            g_loss = bce(d_fake, real)
            q_loss_g = (
                ce(q_fake[:, :cat_dim], cat_idx) +
                mse(q_fake[:, cat_dim:], c_cont)
            )

            optG.zero_grad()
            (g_loss + q_loss_g).backward()
            optG.step()

            total_d += d_loss.item()
            total_g += g_loss.item()
            total_q += q_loss.item()

        print(f"[InfoGAN MLP] Epoch {ep+1}/{epochs} D={total_d/len(train_loader):.4f} G={total_g/len(train_loader):.4f} Q={total_q/len(train_loader):.4f}")

    return G
