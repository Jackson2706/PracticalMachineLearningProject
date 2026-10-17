"""Train a neural classifier with the fixed rules shared by every size:

AdamW (weight decay 0.01), batch 64, at most 30 epochs, early stopping with
patience 5 on validation macro-F1, best checkpoint restored, gradient clipping
at 1.0, cross-entropy loss, fp32 (``amp`` exists for experiments only).
Device handling goes exclusively through :mod:`ksent.hardware`.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ksent.config import global_cfg
from ksent.eval.cost import Timer
from ksent.eval.metrics import macro_f1
from ksent.features.tokenize import Tokens
from ksent.features.vocab import Vocab
from ksent.hardware import (AcceleratorMemoryTracker, assert_on_device, autocast, loader_kwargs, record_mps_fallbacks,
                            seed_everything, to_device)
from ksent.models.registry import build_neural, count_params, width_for
from ksent.pipeline import NeuralPipeline


def fit_neural(method: str, params: dict[str, Any], train_docs: list[Tokens], y_train: np.ndarray,
               val_docs: list[Tokens], y_val: np.ndarray, seed: int, device: str, max_len: int,
               tokenizer: str, mamba_backend: str = "torch_ref", max_epochs: int | None = None,
               amp: bool = False, log=print) -> tuple[NeuralPipeline, dict]:
    """Build the vocabulary on ``train_docs``, train on ``device``, restore the best epoch."""
    nc = global_cfg()["neural"]
    max_epochs = int(max_epochs or nc["max_epochs"])
    info: dict[str, Any] = {"mps_fallbacks": []}
    with Timer() as t_vocab:
        vocab = Vocab.build(train_docs)
        ids, lengths = vocab.encode_batch(train_docs, max_len)
    info["featurize_time_s"] = t_vocab.seconds
    info["vocab_size"] = len(vocab)

    seed_everything(seed, device)
    fixed = {k: v for k, v in params.items() if k not in ("lr", "dropout", "width", "config_id")}
    build_kwargs = dict(vocab_size=len(vocab), width=width_for(method, params["width"]),
                        dropout=float(params["dropout"]), max_len=max_len, fixed=fixed,
                        mamba_backend=mamba_backend if method == "small_mamba" else "torch_ref")
    model = to_device(build_neural(method, **build_kwargs), device)   # model.to(device)
    info["model_device"] = assert_on_device(model, device)
    log(f"    model parameters on {info['model_device']}")
    info["params"] = count_params(model)
    info["width"] = build_kwargs["width"]
    info["mamba_backend"] = getattr(model, "backend", None)

    opt = torch.optim.AdamW(model.parameters(), lr=float(params["lr"]), weight_decay=float(nc["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda") if amp else None  # amp: CUDA experiments only
    loss_fn = nn.CrossEntropyLoss()
    ds = TensorDataset(torch.from_numpy(ids), torch.from_numpy(lengths),
                       torch.tensor(np.array(y_train, dtype=np.int64)))
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=int(nc["batch_size"]), shuffle=True, generator=gen, **loader_kwargs(device))
    clip, patience = float(nc["grad_clip"]), int(nc["patience"])
    pipe = NeuralPipeline(method, model, vocab, max_len, build_kwargs, tokenizer, device)
    val_feats = pipe.prepare(val_docs)
    mem = AcceleratorMemoryTracker(device)

    best_f1, best_state, best_epoch, bad = -1.0, None, 0, 0
    history = []
    with Timer(device) as t_train, record_mps_fallbacks(device, info["mps_fallbacks"]):
        for epoch in range(1, max_epochs + 1):
            model.train()
            total = 0.0
            for bx, bl, by in loader:
                bx = to_device(bx[:, : int(bl.max())], device)
                bl, by = to_device(bl, device), to_device(by, device)
                with autocast(device, amp):
                    loss = loss_fn(model(bx, bl), by)
                opt.zero_grad(set_to_none=True)
                if scaler is None:  # fp32 (all reported runs)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                    opt.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                    scaler.step(opt)
                    scaler.update()
                total += float(loss.detach()) * len(by)
                mem.sample()
            pred, _ = pipe.predict_prepared(val_feats)
            f1 = macro_f1(y_val, pred)
            history.append({"epoch": epoch, "train_loss": total / len(ds), "val_macro_f1": f1})
            if f1 > best_f1:
                best_f1, best_epoch, bad = f1, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            log(f"    epoch {epoch:2d} loss={total / len(ds):.4f} val_f1={f1:.4f}{' *' if bad == 0 else ''}")
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    info.update(classifier_time_s=t_train.seconds, best_epoch=best_epoch, epochs_run=len(history),
                best_val_macro_f1=best_f1, history=history, peak_accelerator_mem_train_mb=mem.peak_mb())
    return pipe, info
