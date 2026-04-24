import torch
import torch.nn as nn
from .normalizer import RMSNorm

class GRUGlobal(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.gru_cell = nn.GRU(dim, dim, batch_first=True)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        residual = x
        x = self.norm(x)

        h = torch.zeros(1, B, D, device=x.device, dtype=x.dtype)
        outputs, _ = self.gru_cell(x, h)

        out = self.drop(self.out_proj(outputs))
        return residual + out
