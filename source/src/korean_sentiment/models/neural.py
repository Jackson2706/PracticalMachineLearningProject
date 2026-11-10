from __future__ import annotations

import sys
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def sequence_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
    return torch.arange(width, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


class AverageMLP(nn.Module):
    def __init__(self, vocab_size: int, config: dict[str, Any]) -> None:
        super().__init__()
        embedding_dim = int(config["embedding_dim"])
        hidden_dim = int(config["hidden_dim"])
        dropout = float(config["dropout"])
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        layers: list[nn.Module] = []
        width = embedding_dim
        for _ in range(int(config.get("num_layers", 1))):
            layers.extend([nn.Linear(width, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            width = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(width, 2)

    def encode(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = sequence_mask(lengths, ids.shape[1])
        return self.encoder(masked_mean(self.embedding(ids), mask))

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(ids, lengths))


class RecurrentClassifier(nn.Module):
    def __init__(self, vocab_size: int, config: dict[str, Any], cell: str) -> None:
        super().__init__()
        embedding_dim = int(config["embedding_dim"])
        hidden_dim = int(config["hidden_dim"])
        num_layers = int(config.get("num_layers", 1))
        dropout = float(config["dropout"])
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        recurrent = nn.LSTM if cell == "lstm" else nn.GRU
        self.recurrent = recurrent(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 2)

    def encode(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(self.embedding(ids), lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.recurrent(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=ids.shape[1])
        return self.dropout(masked_mean(output, sequence_mask(lengths, ids.shape[1])))

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(ids, lengths))


class SmallTransformer(nn.Module):
    def __init__(self, vocab_size: int, config: dict[str, Any], max_length: int) -> None:
        super().__init__()
        embedding_dim = int(config["embedding_dim"])
        hidden_dim = int(config["hidden_dim"])
        heads = int(config["num_heads"])
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        dropout = float(config["dropout"])
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.input_projection = nn.Identity() if embedding_dim == hidden_dim else nn.Linear(embedding_dim, hidden_dim)
        self.position = nn.Embedding(max_length, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            hidden_dim,
            heads,
            dim_feedforward=hidden_dim * int(config.get("ff_multiplier", 2)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, int(config.get("num_layers", 2)), enable_nested_tensor=False)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 2)
        self.representation_dim = hidden_dim

    def encode(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = sequence_mask(lengths, ids.shape[1])
        positions = torch.arange(ids.shape[1], device=ids.device)
        values = self.input_projection(self.embedding(ids)) + self.position(positions).unsqueeze(0)
        values = self.encoder(self.dropout(values), src_key_padding_mask=~mask)
        return masked_mean(self.normalization(values), mask)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.encode(ids, lengths)))


class SmallMamba(nn.Module):
    """Thin adapter over the maintained state-spaces/mamba implementation."""

    def __init__(self, vocab_size: int, config: dict[str, Any], max_length: int) -> None:
        super().__init__()
        if sys.platform != "linux" or not torch.cuda.is_available():
            raise RuntimeError("mamba requires Linux, an NVIDIA GPU, CUDA PyTorch, and mamba-ssm")
        try:
            from mamba_ssm import Mamba
        except ImportError as error:
            raise RuntimeError("mamba-ssm is not installed; run setup-cloud and its Mamba smoke check") from error
        embedding_dim = int(config["embedding_dim"])
        hidden_dim = int(config["hidden_dim"])
        dropout = float(config["dropout"])
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.input_projection = nn.Identity() if embedding_dim == hidden_dim else nn.Linear(embedding_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                Mamba(
                    d_model=hidden_dim,
                    d_state=int(config.get("d_state", 16)),
                    d_conv=int(config.get("d_conv", 4)),
                    expand=int(config.get("expand", 2)),
                )
                for _ in range(int(config.get("num_layers", 2)))
            ]
        )
        self.normalizations = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in self.layers])
        self.output_normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 2)

    def encode(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        values = self.input_projection(self.embedding(ids))
        for layer, normalization in zip(self.layers, self.normalizations):
            values = values + layer(normalization(values))
        return masked_mean(self.output_normalization(values), sequence_mask(lengths, ids.shape[1]))

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.encode(ids, lengths)))


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.layers(values), dim=-1)


def build_neural(model: str, vocab_size: int, config: dict[str, Any], max_length: int) -> nn.Module:
    if model == "avg_mlp":
        return AverageMLP(vocab_size, config)
    if model in {"lstm", "gru"}:
        return RecurrentClassifier(vocab_size, config, model)
    if model == "transformer":
        return SmallTransformer(vocab_size, config, max_length)
    if model == "mamba":
        return SmallMamba(vocab_size, config, max_length)
    raise KeyError(model)
