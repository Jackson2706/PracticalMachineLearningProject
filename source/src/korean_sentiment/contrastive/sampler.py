from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class DomainLabelBalancedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        labels: Sequence[int],
        domains: Sequence[int],
        batch_size: int = 48,
        seed: int = 0,
        target_domain: int | None = None,
    ) -> None:
        if len(labels) != len(domains) or not labels:
            raise ValueError("labels and domains must be non-empty and equal length")
        cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, (domain, label) in enumerate(zip(domains, labels)):
            cells[(int(domain), int(label))].append(index)
        self.cells = dict(sorted(cells.items()))
        if len(self.cells) != 6:
            raise ValueError(f"pooled runs require six domain-label cells, found {len(self.cells)}")
        if batch_size % len(self.cells):
            raise ValueError("effective batch size must be divisible by the present domain-label cells")
        self.per_cell = batch_size // len(self.cells)
        if self.per_cell < 2:
            raise ValueError("each present domain-label cell needs at least two examples per batch")
        self.batch_size = batch_size
        self.seed = seed
        self.target_domain = target_domain
        self.steps = max(1, math.ceil(len(labels) / batch_size))
        self.epoch = 0
        self.replacement_counts: Counter[str] = Counter()
        self.target_sample_counts: Counter[int] = Counter()
        self.total_optimizer_steps = 0

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[list[int]]:
        generator = np.random.default_rng([self.seed, self.epoch])
        required = self.steps * self.per_cell
        streams: dict[tuple[int, int], np.ndarray] = {}
        for cell, indices in self.cells.items():
            replace = required > len(indices)
            stream = generator.choice(indices, size=required, replace=replace)
            streams[cell] = stream
            self.replacement_counts[f"{cell[0]}:{cell[1]}"] += int(required - len(set(stream.tolist())))
        for step in range(self.steps):
            batch: list[int] = []
            for cell, stream in streams.items():
                selected = stream[step * self.per_cell : (step + 1) * self.per_cell].tolist()
                batch.extend(selected)
                if self.target_domain is not None and cell[0] == self.target_domain:
                    self.target_sample_counts.update(selected)
            generator.shuffle(batch)
            self.total_optimizer_steps += 1
            yield batch
        self.epoch += 1

    def statistics(self) -> dict:
        return {
            "effective_batch_size": self.batch_size,
            "examples_per_cell": self.per_cell,
            "replacement_counts": dict(self.replacement_counts),
            "optimizer_steps": self.total_optimizer_steps,
            "target_example_sample_counts": dict(sorted(self.target_sample_counts.items())),
        }
