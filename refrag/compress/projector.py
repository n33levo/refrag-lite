"""MLP projector."""
import torch
import torch.nn as nn
from typing import List

class Projector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [512, 1024],
                 activation: str = "gelu", dropout: float = 0.1, layer_norm: bool = True):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.LayerNorm(h) if layer_norm else nn.Identity(),
                          nn.GELU() if activation == "gelu" else nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.projection = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)