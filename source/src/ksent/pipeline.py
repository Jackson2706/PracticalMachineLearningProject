"""Inference pipelines (raw text -> label) used for evaluation, latency and the demo.

``text_to_input`` performs the full preprocessing of one raw review (normalize ->
tokenize, or normalize only for character TF-IDF); ``prepare`` maps a batch of such
inputs to classifier inputs (TF-IDF matrix or padded ids). Latency is therefore
measured classifier-only (prepared input) and end-to-end (raw string -> label).
Checkpoints are device-independent (CPU tensors) and loaded with ``map_location``.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ksent.data.normalize import normalize
from ksent.features.tokenize import get_tokenizer, tfidf_uses_text


class TfidfPipeline:
    """Fitted TF-IDF vectoriser + scikit-learn/XGBoost classifier (always CPU)."""

    kind = "tfidf"
    device = "cpu"

    def __init__(self, method: str, vectorizer, clf, tokenizer: str) -> None:
        self.method, self.vectorizer, self.clf, self.tokenizer = method, vectorizer, clf, tokenizer
        self._tok = None

    def text_to_input(self, raw: str):
        norm = normalize(raw)
        if tfidf_uses_text(self.tokenizer):
            return norm
        self._tok = self._tok or get_tokenizer(self.tokenizer)
        return self._tok(norm)

    def prepare(self, inputs: list):
        return self.vectorizer.transform(inputs)

    def predict_prepared(self, X) -> tuple[np.ndarray, np.ndarray | None]:
        from ksent.models.classical import predict_scores

        return predict_scores(self.method, self.clf, X)

    def predict_inputs(self, inputs: list) -> tuple[np.ndarray, np.ndarray | None]:
        return self.predict_prepared(self.prepare(inputs))

    def predict_one_prepared(self, x) -> int:
        return int(self.clf.predict(x)[0])

    def predict_text(self, raw: str) -> tuple[int, float | None]:
        """End-to-end: raw string -> (label, P(positive) or None)."""
        pred, prob = self.predict_prepared(self.prepare([self.text_to_input(raw)]))
        return int(pred[0]), (None if prob is None else float(prob[0]))

    def score_text(self, raw: str) -> float | None:
        """Signed decision score for classifiers without probabilities (SVM)."""
        X = self.prepare([self.text_to_input(raw)])
        return float(self.clf.decision_function(X)[0]) if hasattr(self.clf, "decision_function") else None

    def save(self, path: str | Path) -> Path:
        path = Path(path).with_suffix(".joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"method": self.method, "vectorizer": self.vectorizer, "clf": self.clf,
                     "tokenizer": self.tokenizer}, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TfidfPipeline":
        d = joblib.load(path)
        return cls(d["method"], d["vectorizer"], d["clf"], d["tokenizer"])


class NeuralPipeline:
    """Vocabulary + PyTorch classifier with right padding and truncation to ``max_len``."""

    kind = "neural"

    def __init__(self, method: str, model, vocab, max_len: int, build_kwargs: dict[str, Any],
                 tokenizer: str, device: str = "cpu") -> None:
        self.method, self.model, self.vocab, self.max_len = method, model, vocab, max_len
        self.build_kwargs, self.tokenizer, self.device = build_kwargs, tokenizer, device
        self._tok = None

    @property
    def mamba_backend(self) -> str | None:
        return getattr(self.model, "backend", None)

    # --- inputs -------------------------------------------------------------
    def text_to_input(self, raw: str) -> list[str]:
        self._tok = self._tok or get_tokenizer(self.tokenizer)
        return self._tok(normalize(raw))

    def prepare(self, docs: list[list[str]]):
        import torch

        from ksent.hardware import to_device

        ids, lengths = self.vocab.encode_batch(docs, self.max_len)
        return to_device(torch.from_numpy(ids), self.device), to_device(torch.from_numpy(lengths), self.device)

    def prepare_each(self, docs: list[list[str]]) -> list:
        """One ``(1, L)`` input per document (batch-size-1 latency)."""
        return [self.prepare([d]) for d in docs]

    # --- prediction ---------------------------------------------------------
    def predict_prepared(self, feats, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
        import torch

        ids, lengths = feats
        self.model.eval()
        preds, probs = [], []
        with torch.inference_mode():
            for i in range(0, len(ids), batch_size):
                b_ids, b_len = ids[i : i + batch_size], lengths[i : i + batch_size]
                b_ids = b_ids[:, : int(b_len.max())]
                p = torch.softmax(self.model(b_ids, b_len).float(), -1)
                preds.append(p.argmax(-1).cpu().numpy())
                probs.append(p[:, 1].cpu().numpy())
        return np.concatenate(preds).astype(int), np.concatenate(probs)

    def predict_inputs(self, docs: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
        """Batch prediction from token lists (sorted by length for less padding)."""
        order = np.argsort([len(d) for d in docs], kind="stable")
        pred, prob = self.predict_prepared(self.prepare([docs[i] for i in order]))
        out_pred, out_prob = np.empty_like(pred), np.empty_like(prob)
        out_pred[order], out_prob[order] = pred, prob
        return out_pred, out_prob

    def predict_one_prepared(self, feat) -> int:
        import torch

        with torch.inference_mode():
            return int(self.model(*feat).argmax(-1).item())

    def predict_text(self, raw: str) -> tuple[int, float]:
        import torch

        with torch.inference_mode():
            p = torch.softmax(self.model(*self.prepare([self.text_to_input(raw)])).float(), -1)[0]
        return int(p.argmax().item()), float(p[1].item())

    # --- devices / persistence ----------------------------------------------
    def to(self, device: str) -> "NeuralPipeline":
        """Copy of this pipeline on ``device``. Mamba keeps its CUDA backend only on CUDA;
        everywhere else (in particular CPU latency) it is rebuilt with ``torch_ref``."""
        from ksent.hardware import is_cuda, to_device
        from ksent.models.registry import build_neural

        kw = copy.deepcopy(self.build_kwargs)
        if kw.get("mamba_backend") == "mamba_ssm" and not is_cuda(device):
            kw["mamba_backend"] = "torch_ref"
        model = build_neural(self.method, **kw)
        model.load_state_dict({k: v.detach().cpu() for k, v in self.model.state_dict().items()})
        model = to_device(model, device).eval()
        return NeuralPipeline(self.method, model, self.vocab, self.max_len, kw, self.tokenizer, device)

    def save(self, path: str | Path) -> Path:
        import torch

        path = Path(path).with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"method": self.method, "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                    "vocab": self.vocab.to_dict(), "max_len": self.max_len, "build_kwargs": self.build_kwargs,
                    "tokenizer": self.tokenizer}, path)
        return path

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "NeuralPipeline":
        import torch

        from ksent.features.vocab import Vocab
        from ksent.hardware import is_cuda, to_device
        from ksent.models.registry import build_neural

        d = torch.load(path, map_location="cpu", weights_only=False)
        kw = d["build_kwargs"]
        if kw.get("mamba_backend") == "mamba_ssm" and not is_cuda(device):
            kw["mamba_backend"] = "torch_ref"
        model = build_neural(d["method"], **kw)
        model.load_state_dict(d["state_dict"])
        model = to_device(model, device).eval()
        return cls(d["method"], model, Vocab.from_dict(d["vocab"]), d["max_len"], kw, d["tokenizer"], device)


def load_pipeline(path: str | Path, device: str = "cpu"):
    """Load either pipeline type from its artefact."""
    path = Path(path)
    return TfidfPipeline.load(path) if path.suffix == ".joblib" else NeuralPipeline.load(path, device)
