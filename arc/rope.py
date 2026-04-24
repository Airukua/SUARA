import torch

def precompute_rope_freqs(head_dim, max_seq, base=10000, device=None):
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    if max_seq <= 0:
        raise ValueError(f"max_seq must be positive, got {max_seq}")
    if base <= 0:
        raise ValueError(f"base must be positive, got {base}")

    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t     = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(t, theta)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(q, k, freqs_cis):
    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got {tuple(q.shape)} and {tuple(k.shape)}")
    if q.dim() != 4:
        raise ValueError(f"expected q and k with shape [batch, heads, seq, dim], got {tuple(q.shape)}")

    def rotate(x):
        B, H, L, D = x.shape
        if D % 2 != 0:
            raise ValueError(f"RoPE requires an even head dimension, got {D}")
        if freqs_cis.size(0) < L or freqs_cis.size(1) != D // 2:
            raise ValueError(
                f"freqs_cis has incompatible shape {tuple(freqs_cis.shape)} for input shape {tuple(x.shape)}"
            )
        xc = x.float().reshape(B, H, L, D // 2, 2)
        xc = torch.view_as_complex(xc)
        xc = xc * freqs_cis[None, None, :L, :]
        return torch.view_as_real(xc).reshape(B, H, L, D).to(x.dtype)
    return rotate(q), rotate(k)
