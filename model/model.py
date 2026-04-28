import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from arc.block import CrystalWaveBlock
from arc.normalizer import RMSNorm

class CrystalWaveModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim=512,
        n_layers=6,
        n_attn_heads=4,
        n_wave_heads=4,
        n_scales=4,
        sigma_scales=None,
        ff_mult=8 / 3,
        dropout=0.1,
        max_seq=512,
        attention_mode="self",
        ffn_mode="moe",
        gradient_checkpointing=False,
        num_experts=8,
        active_experts=2,
        aux_loss_coef=0.01,
    ):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if max_seq <= 0:
            raise ValueError(f"max_seq must be positive, got {max_seq}")

        self.dim = dim
        self.vocab_size = vocab_size
        self.max_seq = max_seq
        self.attention_mode = attention_mode
        self.gradient_checkpointing = gradient_checkpointing
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [
                CrystalWaveBlock(
                    dim=dim,
                    n_attn_heads=n_attn_heads,
                    n_wave_heads=n_wave_heads,
                    n_scales=n_scales,
                    sigma_scales=sigma_scales,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    max_seq=max_seq,
                    attention_mode=attention_mode,
                    ffn_mode=ffn_mode,
                    num_experts=num_experts,
                    active_experts=active_experts,
                    aux_loss_coef=aux_loss_coef,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_final = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self._init_weights()

    def _forward_block(self, block, x):
        if self.training and self.gradient_checkpointing:
            def custom_forward(hidden_states):
                return block(hidden_states)

            return checkpoint(custom_forward, x, use_reentrant=True)
        return block(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, token_ids, labels=None):
        if token_ids.dim() != 2:
            raise ValueError(
                f"token_ids must have shape [batch, seq], got {tuple(token_ids.shape)}"
            )
        if token_ids.size(1) > self.max_seq:
            raise ValueError(
                f"sequence length {token_ids.size(1)} exceeds max_seq {self.max_seq}"
            )
        if token_ids.min().item() < 0 or token_ids.max().item() >= self.vocab_size:
            raise ValueError("token_ids contain values outside the embedding vocabulary range")

        if labels is not None:
            if labels.shape != token_ids.shape:
                raise ValueError(
                    f"labels shape must match token_ids shape, got {tuple(labels.shape)} "
                    f"and {tuple(token_ids.shape)}"
                )
            invalid_mask = (labels != -100) & ((labels < 0) | (labels >= self.vocab_size))
            if invalid_mask.any().item():
                raise ValueError(
                    "labels contain values outside the valid class range or ignore_index=-100"
                )

        x = self.embedding(token_ids)
        aux_loss = x.new_zeros(())
        for block in self.blocks:
            x, block_aux_loss = self._forward_block(block, x)
            aux_loss = aux_loss + block_aux_loss
        x = self.norm_final(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            loss = loss + aux_loss
        return logits, loss
