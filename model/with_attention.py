import torch.nn as nn
import torch.nn.functional as F
from arc.block import CrystalWaveBlock
from arc.normalizer import RMSNorm

class WithAttention(nn.Module):
    def __init__(self, vocab_size, dim=512, n_layers=6,
                 n_attn_heads=4, n_wave_heads=4, n_scales=4,
                 sigma_scales=None, ff_mult=8/3, dropout=0.1, max_seq=512):
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

        self.dim        = dim
        self.vocab_size = vocab_size
        self.max_seq    = max_seq
        self.embedding  = nn.Embedding(vocab_size, dim)
        self.blocks     = nn.ModuleList([
            CrystalWaveBlock(dim, n_attn_heads, n_wave_heads, n_scales,
                             sigma_scales, ff_mult, dropout, max_seq)
            for _ in range(n_layers)
        ])
        self.norm_final = RMSNorm(dim)
        self.lm_head    = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
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
                raise ValueError("labels contain values outside the valid class range or ignore_index=-100")

        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        x      = self.norm_final(x)
        logits = self.lm_head(x)
        loss   = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1), ignore_index=-100
            )
        return logits, loss
