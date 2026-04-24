import torch
import torch.nn as nn
from arc.casusal_wave_conv import CausalWaveConv
from arc.normalizer import RMSNorm
from arc.ffn import SwiGLU

class CrystalWaveBlock(nn.Module):
    def __init__(self, dim, n_wave_heads=4, n_scales=4,
                 sigma_scales=None, ff_mult=8/3, dropout=0.0, max_seq=512):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.norm1 = RMSNorm(dim)
        self.crystal = CausalWaveConv(
            dim, n_wave_heads, n_scales, sigma_scales, max_seq, dropout
        )
        
        self.gate = nn.Linear(dim, dim, bias=False)
        self.gate_drop = nn.Dropout(dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ff_mult, dropout)
        nn.init.normal_(self.gate.weight, std=0.02)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"expected input with shape [batch, seq, dim], got {tuple(x.shape)}")

        residual = x
        x_norm = self.norm1(x)
        c_out = self.crystal(x_norm)
        g = torch.sigmoid(self.gate(x_norm))
        g = self.gate_drop(g)
        x = residual + (g * c_out)
        x = x + self.ffn(self.norm2(x))
        return x
