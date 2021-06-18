import torch
from torch import nn


class SaltAndPepper(nn.Module):
    def __init__(self, quantity: float = 0.10):
        super(SaltAndPepper, self).__init__()
        self.quantity = quantity

    def forward(self, x):
        rnd = torch.rand(x.shape)
        x[rnd < self.quantity / 2] = 0.
        x[rnd > 1 - self.quantity / 2] = 1.
        return x
