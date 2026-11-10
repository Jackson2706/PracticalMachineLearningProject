from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from korean_sentiment.config import experiment_config
from korean_sentiment.features.text import UnicodeWordTokenizer, Vocabulary


class ModelBlocked(RuntimeError):
    pass


class TfidfPipeline:
    family = "tfidf"
    device = "cpu"

    def __init__(self, model_name: str, config: dict[str, Any], seed: int, cpu_threads: int = 1) -> None:
        self.model_name = model_name
        self.config = dict(config)
        self.seed = seed
        tfidf = experiment_config()["tfidf"]
        self.vectorizer = TfidfVectorizer(
            tokenizer=UnicodeWordTokenizer(),
            token_pattern=None,
            lowercase=False,
            ngram_range=tuple(tfidf["ngram_range"]),
            max_features=int(tfidf["max_features"]),
            norm="l2",
            sublinear_tf=bool(tfidf["sublinear_tf"]),
            dtype=np.float32,
        )
        parameters = {key: value for key, value in config.items() if key != "id"}
        if model_name == "tfidf_lr":
            self.estimator = LogisticRegression(random_state=seed, **parameters)
        elif model_name == "tfidf_svm":
            self.estimator = LinearSVC(random_state=seed, **parameters)
        elif model_name == "tfidf_rf":
            self.estimator = RandomForestClassifier(random_state=seed, n_jobs=cpu_threads, **parameters)
        elif model_name == "tfidf_xgb":
            try:
                from xgboost import XGBClassifier
            except ImportError as error:
                raise ModelBlocked("xgboost is not installed") from error
            self.estimator = XGBClassifier(
                random_state=seed,
                n_jobs=cpu_threads,
                tree_method="hist",
                eval_metric="logloss",
                verbosity=0,
                **parameters,
            )
        else:
            raise KeyError(model_name)

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        validation_texts: Sequence[str] | None = None,
        validation_labels: Sequence[int] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        features = self.vectorizer.fit_transform(list(texts))
        self.estimator.fit(features, np.asarray(labels, dtype=np.int64))
        return {"feature_count": int(features.shape[1]), "best_epoch": None, "history": []}

    def prepare(self, texts: Sequence[str]):
        return self.vectorizer.transform(list(texts))

    def predict_prepared(self, features) -> np.ndarray:
        return np.asarray(self.estimator.predict(features), dtype=np.int64)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return self.predict_prepared(self.prepare(texts))

    def decision_scores(self, texts: Sequence[str]) -> np.ndarray | None:
        features = self.prepare(texts)
        if hasattr(self.estimator, "predict_proba"):
            return np.asarray(self.estimator.predict_proba(features))[:, 1]
        if hasattr(self.estimator, "decision_function"):
            return np.asarray(self.estimator.decision_function(features))
        return None

    def metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {"feature_count": len(self.vectorizer.get_feature_names_out())}
        if hasattr(self.estimator, "estimators_"):
            trees = self.estimator.estimators_
            if isinstance(trees, np.ndarray):
                trees = trees.ravel().tolist()
            result["tree_count"] = len(trees)
            result["node_count"] = int(sum(tree.tree_.node_count for tree in trees))
        elif self.model_name == "tfidf_xgb":
            dump = self.estimator.get_booster().get_dump()
            result["tree_count"] = len(dump)
            result["node_count"] = int(sum(tree.count("\n") for tree in dump))
        return result

    def save(self, path: str | Path) -> Path:
        destination = Path(path).with_suffix(".joblib")
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        os.close(fd)
        try:
            joblib.dump(self, temporary)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "TfidfPipeline":
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise TypeError(f"not a {cls.__name__} artifact")
        return value


class NeuralPipeline:
    family = "neural"

    def __init__(
        self,
        model_name: str,
        config: dict[str, Any],
        seed: int,
        device: str,
        max_length: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.config = dict(config)
        self.seed = seed
        self.device = device
        self.max_length = int(max_length or experiment_config()["neural"]["max_sequence_length"])
        self.tokenizer = UnicodeWordTokenizer()
        self.vocabulary: Vocabulary | None = None
        self.model = None
        self.projection = None
        self.training_info: dict[str, Any] = {}

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        validation_texts: Sequence[str],
        validation_labels: Sequence[int],
        **kwargs: Any,
    ) -> dict[str, Any]:
        from korean_sentiment.training.neural import train_neural

        try:
            self.training_info = train_neural(
                self,
                list(texts),
                np.asarray(labels, dtype=np.int64),
                list(validation_texts),
                np.asarray(validation_labels, dtype=np.int64),
                **kwargs,
            )
        except RuntimeError as error:
            if self.model_name == "mamba" and "mamba" in str(error).lower():
                raise ModelBlocked(str(error)) from error
            raise
        return self.training_info

    def _encoded(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        if self.vocabulary is None:
            raise RuntimeError("pipeline is not fitted")
        documents = [self.tokenizer(text) for text in texts]
        return self.vocabulary.encode_many(documents, self.max_length)

    def prepare(self, texts: Sequence[str]):
        import torch

        ids, lengths = self._encoded(texts)
        return torch.from_numpy(ids).to(self.device), torch.from_numpy(lengths).to(self.device)

    def predict_prepared(self, features, batch_size: int = 256) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("pipeline is not fitted")
        ids, lengths = features
        self.model.eval()
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(ids), batch_size):
                logits = self.model(ids[start : start + batch_size], lengths[start : start + batch_size])
                outputs.append(logits.argmax(dim=-1).cpu().numpy())
        return np.concatenate(outputs).astype(np.int64)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return self.predict_prepared(self.prepare(texts))

    def representations(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        import torch

        ids, lengths = self.prepare(texts)
        self.model.eval()
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(ids), batch_size):
                outputs.append(self.model.encode(ids[start : start + batch_size], lengths[start : start + batch_size]).cpu())
        return torch.cat(outputs).numpy()

    def metadata(self) -> dict[str, Any]:
        if self.model is None or self.vocabulary is None:
            return {}
        total = sum(parameter.numel() for parameter in self.model.parameters())
        trainable = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        projection = sum(parameter.numel() for parameter in self.projection.parameters()) if self.projection else 0
        return {
            "vocab_size": len(self.vocabulary),
            "total_parameters": total + projection,
            "trainable_parameters": trainable + projection,
            "deployed_parameter_count": total,
            "projection_parameters": projection,
            "max_sequence_length": self.max_length,
        }

    def save(self, path: str | Path) -> Path:
        import torch

        if self.model is None or self.vocabulary is None:
            raise RuntimeError("pipeline is not fitted")
        destination = Path(path).with_suffix(".pt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "config": self.config,
            "seed": self.seed,
            "max_length": self.max_length,
            "vocabulary": self.vocabulary.to_dict(),
            "model_state": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
            "projection_state": None
            if self.projection is None
            else {key: value.detach().cpu() for key, value in self.projection.state_dict().items()},
            "training_info": self.training_info,
        }
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "NeuralPipeline":
        import torch

        from korean_sentiment.models.neural import ProjectionHead, build_neural

        payload = torch.load(path, map_location="cpu", weights_only=False)
        pipeline = cls(payload["model_name"], payload["config"], payload["seed"], device, payload["max_length"])
        pipeline.vocabulary = Vocabulary.from_dict(payload["vocabulary"])
        pipeline.model = build_neural(pipeline.model_name, len(pipeline.vocabulary), pipeline.config, pipeline.max_length)
        pipeline.model.load_state_dict(payload["model_state"])
        pipeline.model.to(device).eval()
        if payload.get("projection_state"):
            dimensions = [value.shape for key, value in payload["projection_state"].items() if key.endswith("weight")]
            pipeline.projection = ProjectionHead(dimensions[0][1], dimensions[0][0], dimensions[1][0])
            pipeline.projection.load_state_dict(payload["projection_state"])
            pipeline.projection.to(device).eval()
        pipeline.training_info = payload.get("training_info", {})
        return pipeline


def make_pipeline(model: str, config: dict[str, Any], seed: int, device: str = "cpu", cpu_threads: int = 1):
    if model.startswith("tfidf_"):
        return TfidfPipeline(model, config, seed, cpu_threads)
    return NeuralPipeline(model, config, seed, device)


def load_pipeline(path: str | Path, device: str = "cpu"):
    path = Path(path)
    if path.suffix == ".joblib":
        return TfidfPipeline.load(path)
    if path.suffix == ".pt":
        return NeuralPipeline.load(path, device)
    raise ValueError(f"unsupported model artifact: {path}")
