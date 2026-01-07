import torch
from torch.utils.data import Dataset

class TensorLabelDataset(Dataset):
    """
    Wrapper para garantir que labels sejam torch.Tensor
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, torch.tensor(y, dtype=torch.long)
