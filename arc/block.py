import torch
import torch.nn as nn
from .casusal_wave_conv import CausalWaveConv
from .normalizer import RMSNorm
from .ffn import SwiGLU
from .attn import CausalSelfAttention

class CrystalWaveBlock(nn.Module):
    def __init__(self, dim, n_attn_heads=4, n_wave_heads=4, n_scales=4,
                 sigma_scales=None, ff_mult=8/3, dropout=0.0, max_seq=512):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        self.norm1   = RMSNorm(dim)
        self.crystal = CausalWaveConv(
            dim=dim,
            n_wave_heads=n_wave_heads,
            n_scales=n_scales,
            sigma_scales=sigma_scales,
            dropout=dropout,
        )
        self.attn    = CausalSelfAttention(dim, n_attn_heads, dropout, max_seq)
        self.gate_c  = nn.Linear(dim, dim, bias=False)
        self.gate_a  = nn.Linear(dim, dim, bias=False)
        self.norm2   = RMSNorm(dim)
        self.ffn     = SwiGLU(dim, ff_mult, dropout)

        nn.init.normal_(self.gate_c.weight, std=0.02)
        nn.init.normal_(self.gate_a.weight, std=0.02)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"expected input with shape [batch, seq, dim], got {tuple(x.shape)}")

        normed  = self.norm1(x)
        c_out   = self.crystal(normed)
        a_out   = self.attn(normed)
        g       = torch.sigmoid(self.gate_c(c_out) + self.gate_a(a_out))
        merged  = g * c_out + (1 - g) * a_out
        x = x + merged
        x = x + self.ffn(self.norm2(x))
        return x
