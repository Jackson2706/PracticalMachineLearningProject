"""Masking and pooling helpers shared by the neural models."""
from __future__ import annotations

import torch


def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Boolean ``(B, L)`` mask, ``True`` at valid (non-pad) positions."""
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``h (B, L, D)`` over valid positions."""
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp_min(1.0)
