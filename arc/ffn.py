import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, dim, ff_mult=8/3, dropout=0.0):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if ff_mult <= 0:
            raise ValueError(f"ff_mult must be positive, got {ff_mult}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        hidden = int(dim * ff_mult)
        hidden = (hidden + 63) // 64 * 64
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up   = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if x.dim() < 2 or x.size(-1) != self.down.out_features:
            raise ValueError(
                f"expected input with trailing dim {self.down.out_features}, got {tuple(x.shape)}"
            )
        return self.down(self.drop(F.silu(self.gate(x)) * self.up(x)))
