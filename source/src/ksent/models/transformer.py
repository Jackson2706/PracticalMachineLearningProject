"""Small Transformer encoder: token + learned positional embeddings -> N pre-norm layers."""
from __future__ import annotations

import torch
from torch import nn

from ksent.models.pooling import lengths_to_mask, masked_mean


class SmallTransformer(nn.Module):
    """``width`` is ``d_model``; a linear projection maps the 128-d embedding to it."""

    def __init__(self, vocab_size: int, embed_dim: int, width: int, dropout: float, max_len: int,
                 num_layers: int = 2, nhead: int = 4, ff_mult: int = 2, pooling: str = "mean",
                 n_classes: int = 2, **_: object) -> None:
        super().__init__()
        if width % nhead:
            raise ValueError(f"d_model={width} not divisible by nhead={nhead}")
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, width)
        self.proj = nn.Identity() if width == embed_dim else nn.Linear(embed_dim, width)
        layer = nn.TransformerEncoderLayer(width, nhead, dim_feedforward=ff_mult * width, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width, n_classes)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        L = ids.size(1)
        mask = lengths_to_mask(lengths, L)
        pos = torch.arange(L, device=ids.device)
        x = self.proj(self.embedding(ids)) + self.pos_embedding(pos)[None]
        x = self.encoder(self.dropout(x), src_key_padding_mask=~mask)
        return self.head(self.dropout(masked_mean(self.norm(x), mask)))
