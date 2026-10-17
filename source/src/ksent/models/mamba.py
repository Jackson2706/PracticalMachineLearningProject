"""Small Mamba classifier with two interchangeable backends.

* ``torch_ref``: pure-PyTorch selective scan (:class:`MambaMixer`); runs on CUDA, MPS, CPU.
* ``mamba_ssm``: the official fused CUDA kernels (Linux + CUDA only).

``torch_ref`` uses the same parameter names and shapes as ``mamba_ssm.Mamba``
(in_proj, conv1d, x_proj, dt_proj, A_log, D, out_proj), so weights trained with either
backend load into the other; CPU latency is always measured with ``torch_ref``.
Backend resolution lives in :func:`ksent.hardware.resolve_mamba_backend`; the backend
actually used is recorded in every result file.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ksent.models.pooling import lengths_to_mask, masked_mean


class MambaMixer(nn.Module):
    """Selective state-space mixer (Gu & Dao, 2023) in plain PyTorch."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dt_min: float = 1e-3, dt_max: float = 1e-1) -> None:
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state, self.d_conv = d_state, d_conv
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        # initialisation follows the reference implementation
        nn.init.uniform_(self.dt_proj.weight, -self.dt_rank**-0.5, self.dt_rank**-0.5)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x (B, L, D) -> (B, L, D)``; causal, so right padding cannot leak into valid positions."""
        L = x.size(1)
        xs, z = self.in_proj(x).chunk(2, dim=-1)
        xs = F.silu(self.conv1d(xs.transpose(1, 2))[..., :L].transpose(1, 2))
        dt, Bm, Cm = self.x_proj(xs).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                       # (B, L, Di)
        A = -torch.exp(self.A_log.float())                      # (Di, N)
        y = selective_scan(xs, dt, A, Bm, Cm) + xs * self.D
        return self.out_proj(y * F.silu(z))


def selective_scan(u: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Sequential selective scan: ``h_t = exp(dt_t A) h_{t-1} + dt_t B_t u_t``, ``y_t = C_t h_t``."""
    dA = torch.exp(dt.unsqueeze(-1) * A)                         # (B, L, Di, N)
    dBu = dt.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)    # (B, L, Di, N)
    h = torch.zeros_like(dA[:, 0])
    ys = []
    for t in range(u.size(1)):
        h = dA[:, t] * h + dBu[:, t]
        ys.append((h * C[:, t].unsqueeze(1)).sum(-1))
    return torch.stack(ys, dim=1)


class MambaBlock(nn.Module):
    """Pre-norm residual block around a Mamba mixer."""

    def __init__(self, d_model: int, backend: str, **mixer_kw: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        if backend == "mamba_ssm":
            from mamba_ssm import Mamba

            self.mixer = Mamba(d_model=d_model, **mixer_kw)
        else:
            self.mixer = MambaMixer(d_model, **mixer_kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class SmallMamba(nn.Module):
    """Embedding -> projection -> N Mamba blocks -> masked mean pooling -> linear."""

    def __init__(self, vocab_size: int, embed_dim: int, width: int, dropout: float, max_len: int,
                 num_layers: int = 2, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 pooling: str = "mean", backend: str = "torch_ref", n_classes: int = 2, **_: object) -> None:
        super().__init__()
        if backend not in ("torch_ref", "mamba_ssm"):
            raise ValueError(f"unknown mamba backend {backend!r}")
        self.backend = backend
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.proj = nn.Identity() if width == embed_dim else nn.Linear(embed_dim, width)
        self.blocks = nn.ModuleList(
            [MambaBlock(width, backend, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(num_layers)]
        )
        self.norm = nn.RMSNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width, n_classes)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = lengths_to_mask(lengths, ids.size(1))
        x = self.dropout(self.proj(self.embedding(ids)))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.dropout(masked_mean(self.norm(x), mask)))
