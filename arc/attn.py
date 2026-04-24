import torch
import torch.nn as nn
import torch.nn.functional as F
from arc.rope import precompute_rope_freqs, apply_rope
from helper.sanity_check import _check_dim_heads, _check_dropout, _check_input

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0, max_seq: int = 512):
        super().__init__()
        _check_dim_heads(dim, n_heads)
        _check_dropout(dropout)

        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.scale    = self.head_dim ** -0.5
        self.max_seq  = max_seq
        self.qkv      = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.drop     = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq, max_seq, dtype=torch.bool)),
            persistent=False,
        )
        self.register_buffer(
            "freqs_cis",
            precompute_rope_freqs(self.head_dim, max_seq),
            persistent=False,
        )

    def _ensure_seq_len(self, L: int, device: torch.device) -> None:
        if L <= self.max_seq and self.mask.device == device:
            return
        new_max = max(L, self.max_seq)
        self.max_seq   = new_max
        self.mask      = torch.tril(torch.ones(new_max, new_max, dtype=torch.bool, device=device))
        self.freqs_cis = precompute_rope_freqs(self.head_dim, new_max, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_input(x)
        B, L, D = x.shape
        self._ensure_seq_len(L, x.device)

        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rope(q, k, self.freqs_cis)

        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.drop.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * self.scale
            att = att.masked_fill(~self.mask[:L, :L].unsqueeze(0).unsqueeze(0), float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.drop(att)
            out = att @ v

        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)
