import torch
from train.train_cnn import train_cnn

def train_and_evaluate_cnn(
    train_loader,
    test_loader,
    device,
    epochs=5
):
    """
    Treina CNN e retorna acurácia no conjunto de teste
    """
    model = train_cnn(train_loader, device, epochs=epochs)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    return acc
