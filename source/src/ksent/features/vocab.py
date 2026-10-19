"""Neural vocabulary: built on the training subset only.

``min_freq`` (global.yaml: 2), capped at 20,000 most frequent tokens (ties broken
alphabetically for determinism) plus ``<pad>`` (0) and ``<unk>`` (1).
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from ksent.config import global_cfg

PAD, UNK = "<pad>", "<unk>"


class Vocab:
    """Token <-> index mapping with padding/truncation to a fixed length."""

    def __init__(self, itos: list[str]) -> None:
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}
        self.pad_id, self.unk_id = self.stoi[PAD], self.stoi[UNK]

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, docs: Sequence[Sequence[str]], min_freq: int | None = None, cap: int | None = None) -> "Vocab":
        """Build from tokenised training documents."""
        nc = global_cfg()["neural"]
        min_freq = int(nc["min_freq"] if min_freq is None else min_freq)
        cap = int(nc["vocab_cap"] if cap is None else cap)
        counts = Counter(t for d in docs for t in d)
        items = sorted((kv for kv in counts.items() if kv[1] >= min_freq), key=lambda kv: (-kv[1], kv[0]))
        return cls([PAD, UNK] + [t for t, _ in items[:cap]])

    def encode(self, tokens: Sequence[str], max_len: int) -> list[int]:
        """Map tokens to ids and truncate to ``max_len`` (no padding)."""
        ids = [self.stoi.get(t, self.unk_id) for t in tokens[:max_len]]
        return ids or [self.unk_id]  # empty docs become a single <unk>

    def encode_batch(self, docs: Sequence[Sequence[str]], max_len: int) -> tuple[np.ndarray, np.ndarray]:
        """Right-padded id matrix ``(n, L)`` and lengths ``(n,)`` with L = longest doc in batch."""
        enc = [self.encode(d, max_len) for d in docs]
        lengths = np.array([len(e) for e in enc], dtype=np.int64)
        out = np.full((len(enc), int(lengths.max())), self.pad_id, dtype=np.int64)
        for i, e in enumerate(enc):
            out[i, : len(e)] = e
        return out, lengths

    def to_dict(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, d: dict) -> "Vocab":
        return cls(list(d["itos"]))
