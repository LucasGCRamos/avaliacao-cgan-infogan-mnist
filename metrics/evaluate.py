"""
metrics/evaluate.py
Combine feature extraction and metric computations.
"""

import torch
from models.feature_extractor import FeatureExtractorMNIST
from .fid_kid import compute_fid, compute_kid
from .precision_recall import precision_recall


def extract_features(loader, model, device, n_samples=2000):
    model.eval()
    feats = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            f = model(x)
            feats.append(f)
            if len(feats) * x.size(0) >= n_samples:
                break
    feats = torch.cat(feats)[:n_samples]
    return feats


def evaluate_gan(
    generator,
    real_loader,
    device,
    latent_dim=100,
    conditional=False,
    n_samples=2000
):
    feat_model = FeatureExtractorMNIST().to(device)
    
    real_feats = extract_features(
        real_loader, feat_model, device, n_samples=n_samples
    )

    generator.eval()
    with torch.no_grad():

        if conditional:
            labels = torch.randint(0, 10, (n_samples,), device=device)
            z = torch.randn(n_samples, latent_dim, device=device)
            fake = generator(z, labels)

        else:
            z = torch.randn(n_samples, latent_dim, device=device)

            c_cat = torch.zeros(n_samples, 10, device=device)
            c_cat.scatter_(
                1,
                torch.randint(0, 10, (n_samples, 1), device=device),
                1.0
            )

            c_cont = torch.randn(n_samples, 2, device=device) * 0.1

            try:
                # InfoGAN Conv: z, c_cat, c_cont
                fake = generator(z, c_cat, c_cont)
            except TypeError:
                # InfoGAN MLP: z, concat(c_cat, c_cont)
                c = torch.cat([c_cat, c_cont], dim=1)
                fake = generator(z, c)

        if fake.dim() == 2:  # [N, 784]
            fake = fake.view(-1, 1, 28, 28)

        fake_feats = feat_model(fake)

    mu_r = real_feats.mean(0)
    mu_f = fake_feats.mean(0)

    sigma_r = torch.cov(real_feats.T)
    sigma_f = torch.cov(fake_feats.T)

    fid = compute_fid(mu_r, sigma_r, mu_f, sigma_f)
    kid = compute_kid(real_feats, fake_feats)
    prec, rec = precision_recall(real_feats, fake_feats)

    return {
        "FID": fid,
        "KID": kid,
        "Precision": prec,
        "Recall": rec
    }

