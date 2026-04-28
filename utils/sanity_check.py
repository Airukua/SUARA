import torch

def _check_dim_heads(dim: int, n_heads: int) -> None:
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    if n_heads <= 0:
        raise ValueError(f"n_heads must be positive, got {n_heads}")
    if dim % n_heads != 0:
        raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")


def _check_dropout(dropout: float) -> None:
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}")


def _check_input(x: torch.Tensor) -> None:
    if x.dim() != 3:
        raise ValueError(
            f"expected input with shape [batch, seq, dim], got {tuple(x.shape)}"
        )
    if x.size(1) <= 0:
        raise ValueError("sequence length must be positive")