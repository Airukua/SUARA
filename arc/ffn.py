import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional, Literal

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

class MoE(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int = 8,
        active_experts: int = 2, 
        ff_mult: float = 8/3,
        dropout: float = 0.0,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.aux_loss_coef = aux_loss_coef
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLU(dim, ff_mult=ff_mult, dropout=dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor):
        B, S, D = x.shape
        x_flat = x.view(-1, D)

        router_logits = self.router(x_flat) 
        weights, selected_experts = torch.topk(
            F.softmax(router_logits, dim=-1), 
            self.active_experts, 
            dim=-1
        ) 

        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (selected_experts == i).any(dim=-1) 
            if mask.any():
                expert_input = x_flat[mask]
                expert_out = expert(expert_input)
                expert_idx_in_topk = (selected_experts[mask] == i).nonzero(as_tuple=True)[1]
                w = weights[mask].gather(1, expert_idx_in_topk.unsqueeze(1)).squeeze(1)
                output[mask] += w.unsqueeze(1) * expert_out

        router_prob = F.softmax(router_logits, dim=-1).mean(dim=0)  # [num_experts]
        expert_fraction = torch.zeros(
            self.num_experts, device=x.device, dtype=x.dtype
        )
        for i in range(self.num_experts):
            mask = (selected_experts == i).any(dim=-1).float()
            expert_fraction[i] = mask.mean()
        aux_loss = self.num_experts * (router_prob * expert_fraction).sum()
        aux_loss = self.aux_loss_coef * aux_loss
        output = output.view(B, S, D)
        return output, aux_loss
    

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        mode: Literal["dense", "moe"] = "moe",
        ff_mult: float = 8 / 3,
        dropout: float = 0.0,
        num_experts: int = 8,
        active_experts: int = 2,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.mode = mode

        if mode == "dense":
            self.layer = SwiGLU(dim, ff_mult=ff_mult, dropout=dropout)
            self.num_experts = 1
            self.active_experts = 1
        elif mode == "moe":
            self.layer = MoE(
                dim=dim,
                num_experts=num_experts,
                active_experts=active_experts,
                ff_mult=ff_mult,
                dropout=dropout,
                aux_loss_coef=aux_loss_coef,
            )
            self.num_experts = num_experts
            self.active_experts = active_experts
        else:
            raise ValueError(f"mode harus 'dense' atau 'moe', got {mode}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.mode == "dense":
            return self.layer(x), None
        else:
            return self.layer(x)