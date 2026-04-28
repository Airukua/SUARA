import torch
import torch.nn as nn
from .casusal_wave_conv import CausalWaveConv
from .normalizer import RMSNorm
from .ffn import FeedForward
from .attn import CausalGLAAttention, CausalNastarAttention, CausalSelfAttention

class CrystalWaveBlock(nn.Module):
    def __init__(
        self, dim, n_attn_heads=4, n_wave_heads=4, n_scales=4,
        sigma_scales=None, ff_mult=8/3, dropout=0.0, max_seq=512,
        attention_mode="self", ffn_mode="moe", num_experts=8, active_experts=2,
        aux_loss_coef=0.01
    ):
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
        if attention_mode == "self":
            self.attn = CausalSelfAttention(dim, n_attn_heads, dropout, max_seq)
        elif attention_mode == "nastar":
            self.attn = CausalNastarAttention(dim, n_attn_heads, dropout, max_seq)
        elif attention_mode == "gla":
            self.attn = CausalGLAAttention(dim, n_attn_heads, dropout, max_seq)
        elif attention_mode == "disabled":
            self.attn = None
        else:
            raise ValueError(
                f"attention_mode harus 'self', 'nastar', 'gla', atau 'disabled', got {attention_mode}"
            )
        self.gate_c  = nn.Linear(dim, dim, bias=False)
        self.gate_a  = nn.Linear(dim, dim, bias=False) if attention_mode != "disabled" else None
        self.norm2   = RMSNorm(dim)
        self.ffn     = FeedForward(
            dim=dim,
            mode=ffn_mode,
            ff_mult=ff_mult,
            dropout=dropout,
            num_experts=num_experts,
            active_experts=active_experts,
            aux_loss_coef=aux_loss_coef,
        )

        nn.init.normal_(self.gate_c.weight, std=0.02)
        if self.gate_a is not None:
            nn.init.normal_(self.gate_a.weight, std=0.02)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"expected input with shape [batch, seq, dim], got {tuple(x.shape)}")

        normed  = self.norm1(x)
        c_out   = self.crystal(normed)
        if self.attn is not None:
            a_out   = self.attn(normed)
            g       = torch.sigmoid(self.gate_c(c_out) + self.gate_a(a_out))
            merged  = g * c_out + (1 - g) * a_out
        else:
            merged  = c_out
        x = x + merged
        ffn_out, aux_loss = self.ffn(self.norm2(x))
        x = x + ffn_out
        if aux_loss is None:
            aux_loss = x.new_zeros(())
        return x, aux_loss
