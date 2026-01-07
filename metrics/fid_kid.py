
"""
metrics/fid_kid.py
Simplified FID/KID computation using a small feature extractor.
"""
from scipy.linalg import sqrtm
import torch
import numpy as np

def compute_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = sqrtm((sigma1 @ sigma2).cpu().numpy())
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    covmean = torch.tensor(covmean, dtype=torch.float32, device=mu1.device)

    fid = diff.dot(diff) + torch.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def polynomial_kernel(x, y):
    return (x @ y.T / x.size(1) + 1) ** 3

def compute_kid(real_feats, fake_feats):
    K_rr = polynomial_kernel(real_feats, real_feats).mean()
    K_ff = polynomial_kernel(fake_feats, fake_feats).mean()
    K_rf = polynomial_kernel(real_feats, fake_feats).mean()
    return float(K_rr + K_ff - 2*K_rf)
