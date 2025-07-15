import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ParametrizedMatrix(ABC, nn.Module):
    def __init__(self, dim, device="cpu"):
        super().__init__()
        self.dim = dim
        self.device = device
        self.A_raw = nn.Parameter(torch.zeros(dim, dim, device=device))

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def inverse(self):
        pass

    def save_state(self, output_path, filename):
        os.makedirs(output_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_path, filename))
        print(f"State dict saved to {os.path.join(output_path, filename)}")

    def load_state(self, state_path, map_location=None):
        if map_location is None:
            map_location = self.device
        state_dict = torch.load(state_path, map_location=map_location)
        self.load_state_dict(state_dict)
        print(f"State dict loaded from {state_path}")
