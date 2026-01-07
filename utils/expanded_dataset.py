from torch.utils.data import DataLoader, ConcatDataset
from utils.tensor_label_dataset import TensorLabelDataset

def build_expanded_loader(
    real_loader,
    fake_dataset,
    batch_size=128,
    shuffle=True
):
    real_dataset = TensorLabelDataset(real_loader.dataset)
    expanded_dataset = ConcatDataset([real_dataset, fake_dataset])

    return DataLoader(
        expanded_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
