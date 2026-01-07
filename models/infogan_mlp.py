import torch.nn as nn
import torch

class Generator(nn.Module):
    """
    Gerador InfoGAN (MLP)
    Entrada: z (ruído) + c (código latente controlável)
    """
    def __init__(self, z_dim=64, c_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + c_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z, c):
        return self.net(torch.cat([z, c], 1))


class DiscriminatorQ(nn.Module):
    """
    Discriminador + Rede Q (estimativa do código c)
    """
    def __init__(self, c_dim=10):
        super().__init__()

        # Parte compartilhada
        self.shared = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2)
        )

        # Saída do discriminador
        self.disc = nn.Linear(256, 1)

        # Saída da rede Q (estimativa de c)
        self.q = nn.Linear(256, c_dim)

    def forward(self, x):
        f = self.shared(x)
        return self.disc(f), self.q(f)
