"""Embedding -> masked mean pooling -> MLP (deep averaging network)."""
from __future__ import annotations

import torch
from torch import nn

from ksent.models.pooling import lengths_to_mask, masked_mean


class EmbMLP(nn.Module):
    """DAN-style classifier; ``width`` is the hidden size of each MLP layer."""

    def __init__(self, vocab_size: int, embed_dim: int, width: int, dropout: float, max_len: int,
                 n_hidden_layers: int = 2, activation: str = "relu", n_classes: int = 2, **_: object) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]
        layers: list[nn.Module] = [nn.Dropout(dropout)]
        d_in = embed_dim
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(d_in, width), act(), nn.Dropout(dropout)]
            d_in = width
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(d_in, n_classes)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = lengths_to_mask(lengths, ids.size(1))
        return self.head(self.mlp(masked_mean(self.embedding(ids), mask)))
