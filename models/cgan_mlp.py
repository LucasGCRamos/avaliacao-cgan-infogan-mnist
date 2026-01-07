import torch.nn as nn
import torch

class Generator(nn.Module):
    """
    Gerador da cGAN usando MLP
    Recebe: vetor aleatório Z + embedding do rótulo
    Retorna: imagem 784 dimensões normalizada para [-1,1]
    """
    def __init__(self, z_dim=64, num_classes=10):
        super().__init__()

        # Embedding transforma rótulo em vetor numérico
        self.embed = nn.Embedding(num_classes, num_classes)

        # Rede MLP
        self.net = nn.Sequential(
            nn.Linear(z_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        l = self.embed(labels)
        x = torch.cat([z, l], 1)
        return self.net(x)


class Discriminator(nn.Module):
    """
    Discriminador da cGAN usando MLP
    Recebe: Imagem achatada + embedding do rótulo
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.embed = nn.Embedding(num_classes, num_classes)

        self.net = nn.Sequential(
            nn.Linear(784 + num_classes, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x, labels):
        l = self.embed(labels)
        x = torch.cat([x, l], 1)
        return self.net(x)
