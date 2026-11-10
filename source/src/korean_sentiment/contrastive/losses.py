from __future__ import annotations

import torch


def supervised_contrastive_loss(
    projections: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor | None = None,
    temperature: float = 0.07,
    cross_domain_weight: float = 1.0,
) -> torch.Tensor:
    if projections.ndim != 2 or labels.ndim != 1 or len(projections) != len(labels):
        raise ValueError("expected projections [batch, dim] and labels [batch]")
    if temperature <= 0 or cross_domain_weight < 1:
        raise ValueError("temperature must be positive and cross-domain weight must be at least one")
    projections = torch.nn.functional.normalize(projections, dim=-1)
    logits = projections @ projections.T / temperature
    identity = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    denominator_mask = ~identity
    masked_logits = logits.masked_fill(~denominator_mask, -torch.inf)
    log_probabilities = logits - torch.logsumexp(masked_logits, dim=1, keepdim=True)
    positives = labels[:, None].eq(labels[None, :]) & denominator_mask
    weights = positives.to(projections.dtype)
    if domains is not None and cross_domain_weight != 1:
        different_domain = domains[:, None].ne(domains[None, :])
        weights = weights * torch.where(
            different_domain,
            torch.as_tensor(cross_domain_weight, device=weights.device, dtype=weights.dtype),
            torch.ones((), device=weights.device, dtype=weights.dtype),
        )
    totals = weights.sum(dim=1)
    valid = totals > 0
    if not valid.any():
        return projections.sum() * 0.0
    normalized = weights[valid] / totals[valid, None]
    return -(normalized * log_probabilities[valid]).sum(dim=1).mean()
