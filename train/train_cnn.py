
"""
train/train_cnn.py
Training loop for CNN classifier.
"""
import torch
import torch.nn as nn
from torch import optim
from models.cnn import CNNClassifier

def train_cnn(train_loader, device, epochs=5, lr=1e-3):
    model = CNNClassifier().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"[CNN] Epoch {ep+1}/{epochs} loss={total_loss/len(train_loader):.4f}")
    return model
