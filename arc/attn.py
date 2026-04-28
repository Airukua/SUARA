import torch
import torch.nn as nn
import torch.nn.functional as F
from arc.rope import precompute_rope_freqs, apply_rope
from utils.sanity_check import _check_dim_heads, _check_dropout, _check_input

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
    
class NastarLinear(nn.Module):
    def forward(self, Q, K, V, eps=1e-6):
        Q = Q / (Q.norm(dim=-1, keepdim=True) + eps)
        K = K / (K.norm(dim=-1, keepdim=True) + eps)

        Q_phi = F.relu(Q) + eps
        K_phi = F.relu(K) + eps

        # Causal prefix sums
        k_phi_v = torch.einsum("bhmd,bhme->bhmde", K_phi, V)      # b h m d e
        prefix_kv = torch.cumsum(k_phi_v, dim=2)
        prefix_z = torch.cumsum(K_phi, dim=2)                     # b h m d
        numer = torch.einsum("bhmd,bhmde->bhme", Q_phi, prefix_kv)
        denom = torch.einsum("bhmd,bhmd->bhm", Q_phi, prefix_z).unsqueeze(-1)
        return numer / (denom + eps)

class CausalNastarAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0, max_seq: int = 512):
        super().__init__()
        _check_dim_heads(dim, n_heads)
        _check_dropout(dropout)

        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.max_seq = max_seq
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.nastar = NastarLinear()

        self.register_buffer(
            "freqs_cis",
            precompute_rope_freqs(self.head_dim, max_seq),
            persistent=False,
        )

    def _ensure_seq_len(self, L: int, device: torch.device) -> None:
        if L <= self.max_seq and self.freqs_cis.device == device:
            return
        new_max = max(L, self.max_seq)
        self.max_seq = new_max
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

        out = self.nastar(q, k, v)
        out = self.drop(out)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class CausalGLAAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.0, max_seq: int = 512, eps: float = 1e-6):
        super().__init__()
        _check_dim_heads(dim, n_heads)
        _check_dropout(dropout)
        self.dim = dim
        self.num_heads = n_heads
        self.head_dim = dim // n_heads
        self.max_seq = max_seq
        self.eps = eps
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.ones(n_heads) * 0.9)

        self.register_buffer(
            "freqs_cis",
            precompute_rope_freqs(self.head_dim, max_seq),
            persistent=False,
        )

    def _ensure_seq_len(self, L: int, device: torch.device) -> None:
        if L <= self.max_seq and self.freqs_cis.device == device:
            return
        new_max = max(L, self.max_seq)
        self.max_seq = new_max
        self.freqs_cis = precompute_rope_freqs(self.head_dim, new_max, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_input(x)
        B, L, D = x.shape
        self._ensure_seq_len(L, x.device)

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, self.freqs_cis)

        g = torch.sigmoid(self.gate(x)).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        q = F.silu(q) + self.eps
        k = F.silu(k) + self.eps

        kv = torch.einsum("bhld,bhle->bhlde", k, v)
        z = k
        kv_gated = self._gated_cumsum(kv, g)
        z_gated = self._gated_cumsum(z, g)
        numer = torch.einsum("bhld,bhlde->bhle", q, kv_gated)
        denom = torch.einsum("bhld,bhld->bhl", q, z_gated).unsqueeze(-1)

        out = numer / (denom + self.eps)
        out = self.drop(out)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)

    def _gated_cumsum(self, x: torch.Tensor, gate: torch.Tensor):
        g_scalar = gate.mean(dim=-1, keepdim=True)
        gamma = self.gamma.view(1, -1, 1, 1)
        g_cumprod = torch.cumprod(gamma * g_scalar, dim=2)
        while g_cumprod.dim() < x.dim():
            g_cumprod = g_cumprod.unsqueeze(-1)
        x_weighted = x * g_cumprod
        return torch.cumsum(x_weighted, dim=2)
