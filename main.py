"""
main.py - orchestration script
Trains CNN baseline, cGAN and InfoGAN (MLP + Conv),
evaluates GAN quality and impact of synthetic data on CNN performance.
"""

import torch
from data import get_mnist_loaders
from train.train_cnn import train_cnn

# GANs
from train.train_cgan import train_cgan
from train.train_infogan import train_infogan
from train.train_cgan_mlp import train_cgan_mlp         
from train.train_infogan_mlp import train_infogan_mlp  

# Metrics & utils
from metrics.evaluate import evaluate_gan
from utils.plotting import save_grid
from utils.synthetic_data import (
    generate_synthetic_cgan,
    generate_synthetic_infogan
)
from utils.expanded_dataset import build_expanded_loader
from utils.train_cnn_scenario import train_and_evaluate_cnn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("Device:", DEVICE)
    train_loader, test_loader = get_mnist_loaders(batch_size=128)

    results = {}

    print("\n== Training CNN baseline (MNIST puro) ==")
    acc_base = train_and_evaluate_cnn(
        train_loader, test_loader, DEVICE, epochs=3
    )
    results["MNIST"] = acc_base

    print("\n== Training cGAN MLP ==")
    G_cgan_mlp = train_cgan_mlp(
        train_loader, DEVICE, latent_dim=100, epochs=5
    )

    z = torch.randn(64, 100, device=DEVICE)
    labels = torch.arange(64, device=DEVICE) % 10
    fake = G_cgan_mlp(z, labels)
    save_grid(fake.view(-1, 1, 28, 28).cpu(),
              "outputs/fake_cgan_mlp.png", nrow=8)

    metrics = evaluate_gan(
        G_cgan_mlp, train_loader, DEVICE,
        latent_dim=100, conditional=True, n_samples=2000
    )
    print("cGAN MLP metrics:", metrics)

    fake_ds = generate_synthetic_cgan(
        G_cgan_mlp, DEVICE, n_per_class=2000, latent_dim=100
    )
    loader = build_expanded_loader(train_loader, fake_ds)
    acc = train_and_evaluate_cnn(loader, test_loader, DEVICE, epochs=3)
    results["MNIST + cGAN MLP"] = acc

    print("\n== Training cGAN Conv ==")
    G_cgan = train_cgan(train_loader, DEVICE, latent_dim=100, epochs=5)

    z = torch.randn(64, 100, device=DEVICE)
    labels = torch.arange(64, device=DEVICE) % 10
    fake = G_cgan(z, labels)
    save_grid(fake.cpu(), "outputs/fake_cgan_conv.png", nrow=8)

    metrics = evaluate_gan(
        G_cgan, train_loader, DEVICE,
        latent_dim=100, conditional=True, n_samples=2000
    )
    print("cGAN Conv metrics:", metrics)

    fake_ds = generate_synthetic_cgan(
        G_cgan, DEVICE, n_per_class=2000, latent_dim=100
    )
    loader = build_expanded_loader(train_loader, fake_ds)
    acc = train_and_evaluate_cnn(loader, test_loader, DEVICE, epochs=3)
    results["MNIST + cGAN Conv"] = acc

    print("\n== Training InfoGAN MLP ==")
    G_info_mlp = train_infogan_mlp(
        train_loader, DEVICE,
        latent_dim=62, cat_dim=10, cont_dim=2, epochs=5
    )

    z = torch.randn(64, 62, device=DEVICE)
    labels = torch.arange(64, device=DEVICE) % 10
    c_cat = torch.zeros(64, 10, device=DEVICE)
    c_cat[torch.arange(64), labels] = 1
    c_cont = torch.zeros(64, 2, device=DEVICE)

    c = torch.cat([c_cat, c_cont], dim=1)
    fake = G_info_mlp(z, c)

    save_grid(fake.view(-1, 1, 28, 28).cpu(),
              "outputs/fake_infogan_mlp.png", nrow=8)

    metrics = evaluate_gan(
        G_info_mlp, train_loader, DEVICE,
        latent_dim=62, conditional=False, n_samples=2000
    )
    print("InfoGAN MLP metrics:", metrics)

    fake_ds = generate_synthetic_infogan(
        G_info_mlp, DEVICE, n_per_class=2000, latent_dim=62
    )
    loader = build_expanded_loader(train_loader, fake_ds)
    acc = train_and_evaluate_cnn(loader, test_loader, DEVICE, epochs=3)
    results["MNIST + InfoGAN MLP"] = acc

    print("\n== Training InfoGAN Conv ==")
    G_info = train_infogan(
        train_loader, DEVICE,
        latent_dim=62, cat_dim=10, cont_dim=2, epochs=5
    )

    fake = G_info(z, c_cat, c_cont)
    save_grid(fake.cpu(), "outputs/fake_infogan_conv.png", nrow=8)

    metrics = evaluate_gan(
        G_info, train_loader, DEVICE,
        latent_dim=62, conditional=False, n_samples=2000
    )
    print("InfoGAN Conv metrics:", metrics)

    fake_ds = generate_synthetic_infogan(
        G_info, DEVICE, n_per_class=2000, latent_dim=62
    )
    loader = build_expanded_loader(train_loader, fake_ds)
    acc = train_and_evaluate_cnn(loader, test_loader, DEVICE, epochs=3)
    results["MNIST + InfoGAN Conv"] = acc

    print("\n== Final comparison (CNN accuracy) ==")
    for k, v in results.items():
        print(f"{k:25s}: {v:.4f}")

    print("\nAll outputs saved to outputs/")

if __name__ == "__main__":
    main()
