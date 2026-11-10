from __future__ import annotations

import numpy as np
import pytest
import torch

from korean_sentiment.contrastive.diagnostics import label_savings, normalized_log_auc, representation_diagnostics
from korean_sentiment.contrastive.losses import supervised_contrastive_loss
from korean_sentiment.contrastive.sampler import DomainLabelBalancedBatchSampler
from korean_sentiment.models.neural import ProjectionHead, SmallTransformer


def _reference(z, labels, domains, temperature, alpha):
    z = torch.nn.functional.normalize(z, dim=-1)
    losses = []
    for i in range(len(z)):
        denominator = torch.logsumexp(torch.stack([z[i] @ z[j] / temperature for j in range(len(z)) if j != i]), 0)
        positives = [j for j in range(len(z)) if j != i and labels[j] == labels[i]]
        if not positives:
            continue
        weights = torch.tensor([alpha if domains[j] != domains[i] else 1.0 for j in positives], dtype=z.dtype)
        weights = weights / weights.sum()
        losses.append(-sum(weight * (z[i] @ z[j] / temperature - denominator) for weight, j in zip(weights, positives)))
    return torch.stack(losses).mean()


@pytest.mark.parametrize("alpha", [1.0, 2.0, 4.0])
def test_supcon_matches_transparent_reference(alpha):
    z = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], requires_grad=True)
    labels = torch.tensor([1, 1, 0, 0])
    domains = torch.tensor([0, 1, 0, 1])
    actual = supervised_contrastive_loss(z, labels, domains, 0.1, alpha)
    expected = _reference(z, labels, domains, 0.1, alpha)
    assert float(actual.detach()) == pytest.approx(float(expected.detach()), rel=1e-6)
    actual.backward()
    assert torch.isfinite(z.grad).all()


def test_supcon_is_stable_for_large_logits_and_skips_missing_positive():
    z = torch.tensor([[1000.0, 0.0], [0.0, 1000.0], [1.0, 1.0]], requires_grad=True)
    labels = torch.tensor([0, 1, 1])
    loss = supervised_contrastive_loss(z, labels, torch.tensor([0, 1, 2]), 0.001, 4.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(z.grad).all()


def test_balanced_sampler_composition_determinism_and_replacement_accounting():
    labels = [0, 1] * 3 + [0, 1] * 6
    domains = [0] * 6 + [1] * 6 + [2] * 6
    first = DomainLabelBalancedBatchSampler(labels, domains, batch_size=12, seed=7, target_domain=0)
    second = DomainLabelBalancedBatchSampler(labels, domains, batch_size=12, seed=7, target_domain=0)
    batches_a = list(first)
    batches_b = list(second)
    assert batches_a == batches_b
    for batch in batches_a:
        cells = {(domain, label): 0 for domain in range(3) for label in (0, 1)}
        for index in batch:
            cells[(domains[index], labels[index])] += 1
        assert set(cells.values()) == {2}
    assert first.statistics()["optimizer_steps"] == len(first)


def test_projection_head_is_not_required_by_deployed_classifier():
    config = {"embedding_dim": 16, "hidden_dim": 16, "num_layers": 1, "num_heads": 4, "ff_multiplier": 2, "dropout": 0.1}
    model = SmallTransformer(20, config, 8).eval()
    projection = ProjectionHead(16, 12, 6)
    ids = torch.randint(0, 20, (3, 8))
    lengths = torch.tensor([8, 7, 6])
    with torch.no_grad():
        before = model(ids, lengths)
        projection(model.encode(ids, lengths))
        after = model(ids, lengths)
    assert torch.equal(before, after)


def test_representation_diagnostics_are_deterministic():
    embeddings = np.asarray([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9], [1, 0.1], [0.1, 1]])
    labels = np.asarray([1, 1, 0, 0, 1, 0])
    domains = np.asarray([0, 1, 0, 1, 2, 2])
    first = representation_diagnostics(embeddings, labels, domains, 4, 91)
    second = representation_diagnostics(embeddings, labels, domains, 4, 91)
    assert first == second
    assert first["cross_domain_sentiment_separation_gap"] > 0


def test_label_savings_and_log_auc():
    result = label_savings(
        [200, 500, 1000, 2000],
        [0.60, 0.63, 0.67, 0.70],
        [200, 500, 1000, 2000],
        [0.64, 0.68, 0.70, 0.72],
        {200: 0.64, 500: 0.65, 1000: 0.66, 2000: 0.67},
    )
    assert result["reference_threshold"] == 1000
    assert result["candidate_threshold"] == 500
    assert result["target_training_labels_saved"] == 500
    assert normalized_log_auc([200, 400, 800], [0.5, 0.6, 0.7]) == pytest.approx(0.6)
