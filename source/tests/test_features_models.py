"""Features, neural models, parameter budgets, pipelines and metrics."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ksent.config import NEURAL_METHODS, global_cfg
from ksent.eval.metrics import classification_metrics
from ksent.features.tfidf import make_vectorizer
from ksent.features.vocab import PAD, UNK, Vocab
from ksent.models.registry import _fixed, build_neural, count_params, model_widths

TRAIN = [["좋/VA", "아요/EF", "!!/SF"], ["별로/MAG", "에요/EF", "ㅠㅠ/SW"], ["최고/NNG", "ㅋㅋ/SW"],
         ["최악/NNG", "이/VCP", "다/EF"], ["좋/VA", "다/EF"]]


def test_tfidf_keeps_one_char_tokens_and_fits_on_train_only() -> None:
    vec = make_vectorizer("kiwi")
    X = vec.fit_transform(TRAIN)
    vocab = set(vec.vocabulary_)
    assert "다/EF" in vocab and "좋/VA 아요/EF" in vocab
    assert vec.transform([["처음/NNG", "보는/VV"]]).nnz == 0  # unseen tokens contribute nothing
    assert np.allclose(np.sqrt(X.multiply(X).sum(1)), 1.0)


def test_char_tfidf_uses_text() -> None:
    vec = make_vectorizer("char").fit(["좋아요 ㅋㅋ", "별로에요 ㅠㅠ"])
    assert "ㅋㅋ" in vec.vocabulary_ and "좋아요" in vec.vocabulary_
    assert all(2 <= len(k) <= 4 for k in vec.vocabulary_)


def test_vocab_min_freq_and_cap() -> None:
    v = Vocab.build(TRAIN)  # config min_freq = 2 -> only 좋/VA and 다/EF survive
    assert global_cfg()["neural"]["min_freq"] == 2
    assert v.itos == [PAD, UNK, "다/EF", "좋/VA"]
    ids, lengths = v.encode_batch([["좋/VA"], ["없는", "토큰", "들"]], max_len=2)
    assert ids.shape == (2, 2) and lengths.tolist() == [1, 2]  # truncation keeps the beginning
    assert ids[0, 1] == v.pad_id and ids[1, 0] == v.unk_id


@pytest.mark.parametrize("method", NEURAL_METHODS)
def test_neural_forward_and_padding(method: str) -> None:
    torch.manual_seed(0)
    m = build_neural(method, vocab_size=50, width=model_widths()[method]["S"], dropout=0.0, max_len=16,
                     fixed=_fixed(method)).eval()
    ids = torch.randint(2, 50, (3, 10))
    lengths = torch.tensor([10, 6, 1])
    ids[1, 6:] = 0
    ids[2, 1:] = 0
    out = m(ids, lengths)
    assert out.shape == (3, 2) and torch.isfinite(out).all()
    short = m(ids[1:2, :6], lengths[1:2])  # right padding must not change a prediction
    assert torch.allclose(out[1], short[0], atol=1e-5)


def test_param_budgets_within_tolerance() -> None:
    nc = global_cfg()["neural"]
    tol = float(nc["param_tolerance"])
    for method in NEURAL_METHODS:
        for lvl, target in nc["width_targets"].items():
            m = build_neural(method, 100, model_widths()[method][lvl], 0.1, 64, _fixed(method))
            n = count_params(m)["non_embedding"]
            assert abs(n / target - 1) <= tol, (method, lvl, n)


def test_pipeline_roundtrip(tmp_path) -> None:
    from ksent.models.classical import build_classifier
    from ksent.pipeline import NeuralPipeline, TfidfPipeline, load_pipeline

    y = np.array([1, 0, 1, 0, 1])
    vec = make_vectorizer("whitespace")
    docs = [d for d in (["좋아요", "!!"], ["별로", "ㅠㅠ"], ["최고", "ㅋㅋ"], ["최악", "이다"], ["좋다"])]
    clf = build_classifier("tfidf_lr", {"C": 1.0}, seed=0, n_threads=1).fit(vec.fit_transform(docs), y)
    tp = TfidfPipeline("tfidf_lr", vec, clf, "whitespace")
    loaded = load_pipeline(tp.save(tmp_path / "lr"))
    assert loaded.predict_text("최고 ㅋㅋㅋㅋ") == tp.predict_text("최고 ㅋㅋㅋㅋ")

    vocab = Vocab.build(TRAIN, min_freq=1)
    kw = dict(vocab_size=len(vocab), width=model_widths()["small_mamba"]["S"], dropout=0.1, max_len=16,
              fixed=_fixed("small_mamba"), mamba_backend="torch_ref")
    model = build_neural("small_mamba", **kw).eval()
    npipe = NeuralPipeline("small_mamba", model, vocab, 16, kw, "kiwi")
    loaded = load_pipeline(npipe.save(tmp_path / "mamba"))
    a, b = npipe.predict_text("별로에요 ㅠㅠ"), loaded.predict_text("별로에요 ㅠㅠ")
    assert a[0] == b[0] and abs(a[1] - b[1]) < 1e-6


def test_metrics() -> None:
    m = classification_metrics([0, 0, 1, 1, 1], [0, 1, 1, 1, 0])
    assert m["confusion"] == [[1, 1], [1, 2]]
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["macro_f1"] == pytest.approx((0.5 + 2 / 3) / 2)
