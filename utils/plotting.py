
"""
utils/plotting.py
Simple helpers to save image grids.
"""
import torch
from torchvision.utils import save_image
import os

def save_grid(tensor, filename, nrow=8):
    # expects tensor in [-1,1]
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    save_image((tensor + 1) / 2.0, filename, nrow=nrow)
