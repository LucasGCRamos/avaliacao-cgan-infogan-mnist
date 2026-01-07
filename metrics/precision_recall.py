
"""
metrics/precision_recall.py
Precision/Recall approximation using k-NN.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

def precision_recall(real_feats, fake_feats, k=3):
    real = real_feats.cpu().numpy()
    fake = fake_feats.cpu().numpy()
    nn_r = NearestNeighbors(n_neighbors=k).fit(real)
    nn_f = NearestNeighbors(n_neighbors=k).fit(fake)
    dist_r, _ = nn_r.kneighbors(real)
    dist_f, _ = nn_f.kneighbors(fake)
    thr_r = dist_r[:, -1].mean()
    thr_f = dist_f[:, -1].mean()
    dist_f2r, _ = nn_r.kneighbors(fake)
    precision = (dist_f2r[:,0] < thr_r).mean()
    dist_r2f, _ = nn_f.kneighbors(real)
    recall = (dist_r2f[:,0] < thr_f).mean()
    return float(precision), float(recall)
