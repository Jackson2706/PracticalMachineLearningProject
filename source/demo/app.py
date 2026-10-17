"""Gradio demo: type a Korean review, pick a dataset, compare all nine models.

Shows the normalized text, the Kiwi tokens, every model's prediction, P(positive) where
available (the SVM shows its signed decision score) and the end-to-end latency measured
live (median of 5 calls). Loads the n=8,000 / seed-13 main-tokenizer artefacts saved by
the main sweep (``results/models/{dataset}/{method}/n8000_s13.{joblib,pt}``).
Runs on CPU or any accelerator (neural models only; TF-IDF models always run on CPU).

Usage:
  python demo/app.py [--device cpu|auto|cuda|mps] [--size 8000] [--seed 13] [--port 7860] [--share]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ksent  # noqa: E402,F401  (imports xgboost before torch - see ksent/__init__.py)
import pandas as pd  # noqa: E402

from ksent.config import DATASET_DISPLAY, METHOD_DISPLAY, family, global_cfg, path  # noqa: E402
from ksent.data.normalize import normalize  # noqa: E402
from ksent.features.tokenize import get_tokenizer  # noqa: E402
from ksent.hardware import resolve_device, set_threads, synchronize  # noqa: E402
from ksent.pipeline import load_pipeline  # noqa: E402

EXAMPLES = [
    ["배송도 빠르고 품질도 좋아요 강추합니다!!!!", "shopping"],
    ["돈 아깝다 ㅠㅠㅠ 두 번 다시 안 봄", "nsmc"],
    ["스토리는 좋은데 최적화가 너무 엉망이네요", "steam"],
    ["ㅋㅋㅋㅋㅋ 인생 영화 10점 만점에 10점", "nsmc"],
]
_CACHE: dict[tuple[str, str], dict[str, object]] = {}


def artefact(dataset: str, method: str, size: int, seed: int) -> Path | None:
    base = path("results") / "models" / dataset / method / f"n{size}_s{seed}"
    for ext in (".joblib", ".pt"):
        if base.with_suffix(ext).exists():
            return base.with_suffix(ext)
    return None


def load_all(dataset: str, size: int, seed: int, device: str) -> dict[str, object]:
    key = (dataset, device)
    if key not in _CACHE:
        pipes = {}
        for m in global_cfg()["methods"]:
            p = artefact(dataset, m, size, seed)
            if p is not None:
                pipes[m] = load_pipeline(p, device="cpu" if family(m) == "tfidf" else device)
        _CACHE[key] = pipes
    return _CACHE[key]


def predict(text: str, dataset: str, size: int, seed: int, device: str = "cpu") -> tuple[pd.DataFrame, str]:
    text = (text or "").strip()
    if not text:
        return pd.DataFrame(), "Please enter a review."
    norm = normalize(text)
    tokens = " · ".join(get_tokenizer("kiwi")(norm))
    header = f"**Normalized:** {norm}\n\n**Kiwi tokens:** {tokens}\n\n"
    pipes = load_all(dataset, size, seed, device)
    if not pipes:
        return pd.DataFrame(), header + (f"No saved models for {dataset} (n={size}, seed={seed}). "
                                         "Run scripts/05_run_all.sh first.")
    rows = []
    for m in global_cfg()["methods"]:
        fam = "TF-IDF" if family(m) == "tfidf" else "Neural"
        pipe = pipes.get(m)
        if pipe is None:
            rows.append({"Method": METHOD_DISPLAY[m], "Family": fam, "Prediction": "(model missing)"})
            continue
        dev = getattr(pipe, "device", "cpu")
        pipe.predict_text(text)  # warm-up
        times, out = [], None
        for _ in range(5):
            synchronize(dev)
            t0 = time.perf_counter()
            out = pipe.predict_text(text)
            synchronize(dev)
            times.append((time.perf_counter() - t0) * 1e3)
        label, prob = out
        conf = f"P(pos)={prob:.3f}" if prob is not None else f"score={pipe.score_text(text):+.3f}"
        rows.append({"Method": METHOD_DISPLAY[m], "Family": fam, "Prediction": "positive" if label == 1 else "negative",
                     "Confidence": conf, "Device": dev, "Latency (ms, e2e)": round(statistics.median(times), 3)})
    votes = sum(r.get("Prediction") == "positive" for r in rows)
    return pd.DataFrame(rows), header + f"{votes}/{len(rows)} models predict positive."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu", help="device for neural models (cpu | auto | cuda | mps)")
    ap.add_argument("--size", type=int, default=int(global_cfg()["artifacts"]["save_models_size"]))
    ap.add_argument("--seed", type=int, default=int(global_cfg()["artifacts"]["demo_seed"]))
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    set_threads()
    device, why = resolve_device(args.device, allow_fallback=True)
    print(f"demo device for neural models: {device} ({why})")
    import gradio as gr

    with gr.Blocks(title="Korean Sentiment: TF-IDF vs Neural") as demo:
        gr.Markdown("## Korean review sentiment: nine models side by side\n"
                    f"Models trained on n={args.size:,} reviews (seed {args.seed}). Latency is measured live on "
                    f"this machine (batch size 1, median of 5; neural models on `{device}`).")
        with gr.Row():
            text = gr.Textbox(label="Korean review", lines=3, placeholder="리뷰를 입력하세요")
            ds = gr.Dropdown(choices=[(DATASET_DISPLAY[d], d) for d in global_cfg()["datasets"]],
                             value="nsmc", label="Models trained on")
        btn = gr.Button("Classify", variant="primary")
        info = gr.Markdown()
        table = gr.Dataframe(label="Predictions", interactive=False)
        btn.click(lambda t, d: predict(t, d, args.size, args.seed, device)[::-1], [text, ds], [info, table])
        gr.Examples(EXAMPLES, [text, ds])
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
