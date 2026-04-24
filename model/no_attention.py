import torch.nn as nn
import torch.nn.functional as F
from arc.block import CrystalWaveBlock
from arc.normalizer import RMSNorm

class NoAttention(nn.Module):
    def __init__(self, vocab_size, dim=512, n_layers=6, n_wave_heads=4, n_scales=4,
                 sigma_scales=None, ff_mult=8/3, dropout=0.1, max_seq=512):
        super().__init__()
        self.max_seq = max_seq
        self.embedding  = nn.Embedding(vocab_size, dim)
        self.blocks     = nn.ModuleList([
            CrystalWaveBlock(dim, n_wave_heads, n_scales,
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
        x = self.embedding(token_ids) # -> vector
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
