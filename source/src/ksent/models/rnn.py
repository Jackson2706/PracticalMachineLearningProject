"""Embedding -> (bi)RNN / LSTM / GRU -> masked mean or last-state pooling -> linear."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ksent.models.pooling import lengths_to_mask, masked_mean


class EmbRNN(nn.Module):
    """Recurrent classifier; ``width`` is the hidden size per direction."""

    def __init__(self, vocab_size: int, embed_dim: int, width: int, dropout: float, max_len: int,
                 cell: str = "lstm", num_layers: int = 1, bidirectional: bool = True,
                 pooling: str = "mean", n_classes: int = 2, **_: object) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(embed_dim, width, num_layers=num_layers, batch_first=True,
                           bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        self.cell, self.pooling, self.bidirectional = cell, pooling, bidirectional
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width * (2 if bidirectional else 1), n_classes)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.embedding(ids))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        if self.pooling == "last":
            h_n = h[0] if self.cell == "lstm" else h
            feat = torch.cat([h_n[-2], h_n[-1]], dim=-1) if self.bidirectional else h_n[-1]
        else:
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
            feat = masked_mean(out, lengths_to_mask(lengths, ids.size(1)))
        return self.head(self.dropout(feat))
