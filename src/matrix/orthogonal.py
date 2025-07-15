import torch

from .base import ParametrizedMatrix


class OrthogonalMatrix(ParametrizedMatrix):
    def forward(self):
        anti_sym = self.A_raw - self.A_raw.T
        return torch.matrix_exp(anti_sym)

    def inverse(self):
        M = self()
        return M.T
