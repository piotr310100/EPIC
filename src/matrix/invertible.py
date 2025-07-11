import torch

from .base import ParametrizedMatrix


class InvertibleMatrix(ParametrizedMatrix):
    def forward(self):
        return torch.matrix_exp(self.A_raw)

    def inverse(self):
        M = self()
        return torch.linalg.inv(M)
