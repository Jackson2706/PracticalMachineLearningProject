from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from korean_sentiment.config import env_int, experiment_config
from korean_sentiment.contrastive.losses import supervised_contrastive_loss
from korean_sentiment.contrastive.sampler import DomainLabelBalancedBatchSampler
from korean_sentiment.features.text import Vocabulary
from korean_sentiment.models.neural import ProjectionHead, build_neural
from korean_sentiment.utils.runtime import seed_everything


def _autocast(device: str, precision: str):
    enabled = precision in {"fp16", "bf16"} and device.startswith("cuda")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)


def _predict(model: nn.Module, ids: torch.Tensor, lengths: torch.Tensor, device: str, batch_size: int = 256) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            logits = model(ids[start : start + batch_size].to(device), lengths[start : start + batch_size].to(device))
            predictions.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(predictions)


def train_neural(
    pipeline,
    train_texts: list[str],
    train_labels: np.ndarray,
    validation_texts: list[str],
    validation_labels: np.ndarray,
    domains: np.ndarray | None = None,
    objective: str = "ce",
    contrastive_lambda: float = 0.0,
    contrastive_temperature: float = 0.07,
    cross_domain_weight: float = 1.0,
    target_domain_id: int | None = None,
    effective_batch_size: int | None = None,
    max_epochs: int | None = None,
    patience: int | None = None,
    example_ids: list[str] | None = None,
) -> dict[str, Any]:
    settings = experiment_config()["neural"]
    seed_everything(pipeline.seed)
    documents = [pipeline.tokenizer(text) for text in train_texts]
    validation_documents = [pipeline.tokenizer(text) for text in validation_texts]
    pipeline.vocabulary = Vocabulary.fit(
        documents, cap=int(settings["vocab_cap"]), min_frequency=int(settings["min_frequency"])
    )
    train_ids, train_lengths = pipeline.vocabulary.encode_many(documents, pipeline.max_length)
    validation_ids, validation_lengths = pipeline.vocabulary.encode_many(validation_documents, pipeline.max_length)
    pipeline.model = build_neural(pipeline.model_name, len(pipeline.vocabulary), pipeline.config, pipeline.max_length)
    pipeline.model.to(pipeline.device)
    domains = np.zeros(len(train_labels), dtype=np.int64) if domains is None else np.asarray(domains, dtype=np.int64)
    dataset = TensorDataset(
        torch.from_numpy(train_ids),
        torch.from_numpy(train_lengths),
        torch.from_numpy(train_labels),
        torch.from_numpy(domains),
    )
    sampler = None
    if objective in {"pooled_ce", "supcon", "db_cdsupcon"}:
        sampler = DomainLabelBalancedBatchSampler(
            train_labels.tolist(),
            domains.tolist(),
            batch_size=int(effective_batch_size or 48),
            seed=pipeline.seed,
            target_domain=target_domain_id,
        )
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=env_int("KS_NUM_WORKERS", 0))
    else:
        generator = torch.Generator().manual_seed(pipeline.seed)
        loader = DataLoader(
            dataset,
            batch_size=int(effective_batch_size or settings["batch_size"]),
            shuffle=True,
            generator=generator,
            num_workers=env_int("KS_NUM_WORKERS", 0),
        )
    if objective in {"supcon", "db_cdsupcon"}:
        if pipeline.model_name != "transformer":
            raise ValueError("contrastive objectives require the shared Transformer backbone")
        bonus = __import__("korean_sentiment.config", fromlist=["bonus_config"]).bonus_config()
        pipeline.projection = ProjectionHead(
            pipeline.model.representation_dim,
            int(bonus["projection_hidden_dim"]),
            int(bonus["projection_dim"]),
        ).to(pipeline.device)
    parameters = list(pipeline.model.parameters()) + ([] if pipeline.projection is None else list(pipeline.projection.parameters()))
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(pipeline.config["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    precision = __import__("os").environ.get("KS_PRECISION", "fp32")
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16" and pipeline.device.startswith("cuda"))
    if pipeline.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(pipeline.device)
    criterion = nn.CrossEntropyLoss()
    epochs = int(max_epochs or settings["max_epochs"])
    patience = int(patience or settings["patience"])
    best_score = -1.0
    best_epoch = 0
    best_model = None
    best_projection = None
    bad_epochs = 0
    history = []
    optimizer_steps = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        pipeline.model.train()
        if pipeline.projection is not None:
            pipeline.projection.train()
        loss_sum = 0.0
        sample_count = 0
        for ids, lengths, labels, batch_domains in loader:
            ids = ids.to(pipeline.device)
            lengths = lengths.to(pipeline.device)
            labels = labels.to(pipeline.device)
            batch_domains = batch_domains.to(pipeline.device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(pipeline.device, precision):
                representations = pipeline.model.encode(ids, lengths)
                logits = pipeline.model.classifier(pipeline.model.dropout(representations)) if hasattr(
                    pipeline.model, "dropout"
                ) else pipeline.model.classifier(representations)
                ce_loss = criterion(logits, labels)
                loss = ce_loss
                if pipeline.projection is not None:
                    projections = pipeline.projection(representations)
                    alpha = cross_domain_weight if objective == "db_cdsupcon" else 1.0
                    contrastive = supervised_contrastive_loss(
                        projections, labels, batch_domains, contrastive_temperature, alpha
                    )
                    loss = ce_loss + float(contrastive_lambda) * contrastive
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(parameters, float(settings["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1
            loss_sum += float(loss.detach()) * len(labels)
            sample_count += len(labels)
        predictions = _predict(
            pipeline.model,
            torch.from_numpy(validation_ids),
            torch.from_numpy(validation_lengths),
            pipeline.device,
        )
        score = float(f1_score(validation_labels, predictions, average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": loss_sum / sample_count, "validation_macro_f1": score})
        if score > best_score:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            best_model = copy.deepcopy({key: value.detach().cpu() for key, value in pipeline.model.state_dict().items()})
            best_projection = None if pipeline.projection is None else copy.deepcopy(
                {key: value.detach().cpu() for key, value in pipeline.projection.state_dict().items()}
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    pipeline.model.load_state_dict(best_model)
    pipeline.model.eval()
    if pipeline.projection is not None and best_projection is not None:
        pipeline.projection.load_state_dict(best_projection)
        pipeline.projection.eval()
    elapsed = time.perf_counter() - started
    peak_gpu = torch.cuda.max_memory_allocated(pipeline.device) / 1024**2 if pipeline.device.startswith("cuda") else None
    sampler_statistics = sampler.statistics() if sampler else None
    if sampler_statistics and example_ids:
        sampler_statistics["target_example_sample_counts"] = {
            example_ids[int(index)]: count
            for index, count in sampler_statistics["target_example_sample_counts"].items()
        }
    return {
        "history": history,
        "best_epoch": best_epoch,
        "validation_macro_f1": best_score,
        "fit_time_sec": elapsed,
        "optimizer_steps": optimizer_steps,
        "peak_gpu_memory_mb": peak_gpu,
        "batch_sampler": "domain_label_balanced" if sampler else "shuffle",
        "sampler": sampler_statistics,
    }
