from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


class UnicodeWordTokenizer:
    def __call__(self, text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text)


@dataclass
class Vocabulary:
    tokens: list[str]

    PAD = "<pad>"
    UNK = "<unk>"

    def __post_init__(self) -> None:
        self.stoi = {token: index for index, token in enumerate(self.tokens)}
        if self.tokens[:2] != [self.PAD, self.UNK]:
            raise ValueError("vocabulary must begin with <pad>, <unk>")

    @classmethod
    def fit(cls, documents: Sequence[Sequence[str]], cap: int = 20000, min_frequency: int = 1) -> "Vocabulary":
        counts = Counter(token for document in documents for token in document)
        ordered = sorted(
            ((token, count) for token, count in counts.items() if count >= min_frequency),
            key=lambda item: (-item[1], item[0]),
        )
        return cls([cls.PAD, cls.UNK] + [token for token, _ in ordered[: max(0, cap - 2)]])

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, document: Sequence[str], max_length: int) -> tuple[list[int], int]:
        values = [self.stoi.get(token, 1) for token in document[:max_length]] or [1]
        length = len(values)
        return values + [0] * (max_length - length), length

    def encode_many(self, documents: Sequence[Sequence[str]], max_length: int) -> tuple[np.ndarray, np.ndarray]:
        encoded = [self.encode(document, max_length) for document in documents]
        ids = np.asarray([value[0] for value in encoded], dtype=np.int64)
        lengths = np.asarray([value[1] for value in encoded], dtype=np.int64)
        return ids, lengths

    def to_dict(self) -> dict[str, list[str]]:
        return {"tokens": self.tokens}

    @classmethod
    def from_dict(cls, value: dict[str, list[str]]) -> "Vocabulary":
        return cls(list(value["tokens"]))
