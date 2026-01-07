import torch
from torch.utils.data import TensorDataset


def generate_synthetic_cgan(
    generator,
    device,
    n_per_class=2000,
    latent_dim=100,
    num_classes=10
):
    """
    Gera imagens sintéticas rotuladas usando cGAN
    Compatível com MLP e Conv
    """
    generator.eval()
    images = []
    labels = []

    with torch.no_grad():
        for c in range(num_classes):
            z = torch.randn(n_per_class, latent_dim, device=device)
            y = torch.full((n_per_class,), c, device=device, dtype=torch.long)

            fake = generator(z, y)

            if fake.dim() == 2:  # MLP
                fake = fake.view(-1, 1, 28, 28)

            images.append(fake.cpu())
            labels.append(y.cpu())

    X_fake = torch.cat(images)
    y_fake = torch.cat(labels)

    return TensorDataset(X_fake, y_fake)


def generate_synthetic_infogan(
    generator,
    device,
    n_per_class=2000,
    latent_dim=62,
    num_classes=10,
    cont_dim=2
):
    """
    Gera imagens sintéticas rotuladas usando InfoGAN
    Compatível com MLP e Conv
    """
    generator.eval()
    images = []
    labels = []

    with torch.no_grad():
        for c in range(num_classes):
            z = torch.randn(n_per_class, latent_dim, device=device)

            c_cat = torch.zeros(n_per_class, num_classes, device=device)
            c_cat[:, c] = 1.0

            c_cont = torch.randn(n_per_class, cont_dim, device=device) * 0.1

            try:
                # InfoGAN Conv
                fake = generator(z, c_cat, c_cont)
            except TypeError:
                # InfoGAN MLP
                c_all = torch.cat([c_cat, c_cont], dim=1)
                fake = generator(z, c_all)

            if fake.dim() == 2:  # MLP
                fake = fake.view(-1, 1, 28, 28)

            images.append(fake.cpu())
            labels.append(
                torch.full((n_per_class,), c, dtype=torch.long)
            )

    X_fake = torch.cat(images)
    y_fake = torch.cat(labels)

    return TensorDataset(X_fake, y_fake)
