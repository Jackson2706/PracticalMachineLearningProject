# Training Data Thresholds for Cost-Effective Korean Sentiment Analysis (source)

When does the best neural sequence model (trained from scratch) beat the best TF-IDF
model by ≥ 1 macro-F1 point on Korean reviews, and is it still competitive in cost?

| Experiment | Fits |
|---|---|
| Main sweep: 3 datasets (NSMC, Naver Shopping, Korean Steam) × 9 methods × 5 sizes (500 … 8,000) × 5 seeds | **675** |
| Tuning: 3 × 9 × 10 predefined configs (n=2,000, seed 0, Kiwi) | **270** |
| Tokenizer ablation: best TF-IDF + best neural per dataset × {eojeol, Kiwi, char} × {500, 8,000} × 5 seeds (60 Kiwi cells reused) | **120** new |

Every number in the report and slides is generated from `results/` by `scripts/07_analyze.sh`.

## 1. Setup (conda env `ML`, never a new env)

Install PyTorch for your hardware **first**. `00_setup_env.sh` checks for it and prints
the right command if it is missing:

| Hardware | PyTorch install (inside `conda activate ML`) |
|---|---|
| NVIDIA GPU (Linux/Windows) | `pip install torch --index-url https://download.pytorch.org/whl/cu124` (choose the `cuXXX` matching your driver) |
| Apple Silicon (MPS) | `pip install torch` |
| CPU only | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |

```bash
cd source
bash scripts/00_setup_env.sh   # pinned requirements, mamba-ssm on Linux+CUDA, import checks, results/env/
python -m ksent.hardware       # prints detected hardware and the resolved device
```

`00_setup_env.sh` also handles two platform specifics:
* **Linux + NVIDIA:** it tries `pip install mamba-ssm causal-conv1d`. A failure is tolerated and the `torch_ref` backend is used instead.
* **macOS:** it links `$CONDA_PREFIX/lib/libomp.dylib` to torch's bundled `libomp`. Two different OpenMP runtimes in one process make XGBoost or torch segfault (exit 139).

## 2. Run

```bash
bash scripts/run_everything.sh            # 01 → 08 (+ pytest)
```

or phase by phase:

| Step | Command | Output |
|---|---|---|
| Data | `bash scripts/01_prepare_data.sh` | `data/processed/`, `data/tokenized/`, `results/aggregated/dataset_stats.json`, diagnostics |
| Smoke test | `bash scripts/02_smoke_test.sh` | `results_smoke/cpu/` (27 runs) + `results_smoke/<accelerator>/` (15 neural runs each) |
| Pilot | `bash scripts/03_pilot.sh` | `results_pilot/`, runtime estimate, Mamba backend comparison (CUDA) |
| Tuning | `bash scripts/04_tune.sh` | `results/tuning/`, `configs/frozen/` (**commit them**) |
| Main sweep | `nohup bash scripts/05_run_all.sh > logs/run_all.nohup 2>&1 &` | `results/raw/`, `results/predictions/`, `results/models/` |
| Ablation | `bash scripts/06_run_ablation.sh` | 120 more files under `results/raw/{d}/{m}/{whitespace,char}/` |
| Analysis | `bash scripts/07_analyze.sh` | `results/aggregated/*`, report tables/figures/`numbers.tex` |
| PDFs | `bash scripts/08_build_submission.sh` | `../submission/report/main.pdf`, `../submission/slides/slides.pdf` |
| Tests | `python -m pytest -q` | CUDA/MPS tests skip when the hardware is absent |
| Demo | `python demo/app.py --device auto` | Gradio UI (needs the n=8,000 models) |

Measured on the development machine (Apple M4, 8 threads): data preparation ~40 s; smoke
test ~12 min on CPU + MPS (pure-PyTorch Mamba on CPU dominates). Run `03_pilot.sh` on the
final machine for tuning/sweep/ablation estimates.

### Device and thread control

* `configs/global.yaml: device: auto` resolves to CUDA → MPS → CPU, and the chosen device and the reason are logged. Override with `--device cuda|cuda:N|mps|cpu` on `ksent.run_one`, `ksent.sweep` or the demo, or with `KSENT_DEVICE=...` for the scripts.
* If you request a device that isn't available, the run fails with a clear message. `--allow-fallback` exists for smoke tests only and is refused for final runs.
* `n_threads: 8` is capped at `os.cpu_count()` and can be overridden with `KSENT_THREADS`. Scripts set `CUDA_VISIBLE_DEVICES=0` unless it is already set (one GPU only).
* Examples:
  ```bash
  KSENT_DEVICE=cuda bash scripts/05_run_all.sh                 # NVIDIA
  KSENT_DEVICE=mps  bash scripts/05_run_all.sh                 # Apple Silicon
  KSENT_DEVICE=cpu KSENT_THREADS=16 bash scripts/05_run_all.sh # CPU-only server
  bash scripts/05_run_all.sh --dataset nsmc --method tfidf_lr emb_gru --seeds 13 42
  python -m ksent.sweep --mode final --dry-run                 # list what is left
  python -m ksent.analysis.check_results [--ablation]          # all present + one hardware fingerprint?
  ```
* **Mamba backend:** `mamba_backend: auto` uses `mamba_ssm` on CUDA when importable, else `torch_ref` (pure PyTorch, any device). The two backends share parameter names, and a CUDA test checks that their outputs match (tolerance 1e-4).
  * Pick **one** backend at the pilot and set it explicitly before tuning. All reported Mamba results must share it, and `aggregate.py` warns if they don't.
  * CPU latency always uses `torch_ref`.
* **Precision:** fp32 everywhere. `--amp` exists for CUDA experiments only and is refused for final runs.
* **Determinism:**
  * CUDA uses cuDNN deterministic mode, `use_deterministic_algorithms(warn_only=True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. cuDNN RNNs may still be slightly nondeterministic, and this is recorded in each run.
  * MPS is best-effort. The `PYTORCH_ENABLE_MPS_FALLBACK=1` CPU fallbacks are recorded per run (`mps_fallback_used`).

All device-specific code lives in `src/ksent/hardware.py`.

## 3. Protocol (what the code guarantees)

* **Normalization** (`data/normalize.py`, fixed rules applied before splitting):
  1. `html.unescape`
  2. **NFC** (not NFKC)
  3. strip zero-width and control characters
  4. lowercase Latin letters
  5. cap runs of ≥3 identical characters to 2
  6. collapse whitespace

  Digits, emoji, jamo (ㅋㅋ, ㅠㅠ), English and profanity are kept. There is no spacing correction, stopword removal or digit replacement.
* **Cleaning** (on normalized text): drop empty texts, drop all copies of texts with conflicting labels, keep one copy of exact duplicates, and drop reviews with zero Kiwi tokens. Every step's count is logged. For Shopping, ratings 1–2 → 0 and 4–5 → 1; the number of rating-3 reviews is logged (it is 0).
* **Splits:** data seed 2026; stratified 8,000 / 2,000 / 2,000 per dataset. NSMC pool and validation come from `ratings_train.txt` and test from `ratings_test.txt`. Zero overlap on normalized text is asserted. Raw and normalized text are both stored.
* **Nested subsets:** one stratified ordering per seed (tuning `0`; experiment `13, 42, 87, 2024, 31337`), and sizes are prefixes. Every method and tokenizer reads the same IDs. At n=8,000 each seed uses the full pool, so std there reflects model randomness only.
* **Tokenizers** (`features/tokenize.py`, cached in `data/tokenized/{dataset}/{tokenizer}.parquet`):
  * `kiwi` (main): `form/TAG` morphemes, all particles and endings kept, typo correction off (`kiwipiepy==0.24.0`).
  * `whitespace`: eojeols.
  * `char`: characters for neural models; character 2–4-grams of the normalized text for TF-IDF.
  * A symbol-survival check (ㅋ ㅠ ㅎ emoji) fails loudly if any are dropped.
* **TF-IDF:** pre-tokenized lists with an identity tokenizer and `token_pattern=None`. sklearn's default `\w\w+` would drop 좋/안/못/다, and a test asserts `안/MAG` is in the NSMC n=2,000 vocabulary. Settings: unigrams + bigrams, `max_features=10000`, L2 norm, `sublinear_tf`, `min_df=1`. Always fitted on the current training subset only.
* **Neural:**
  * Vocabulary from the training subset only, `min_freq=2`, capped at 20,000, plus `<pad>` and `<unk>`. Embeddings are 128-d and learned from scratch.
  * `max_seq_len` per tokenizer (kiwi 72, whitespace 32, char 96): the smallest multiple of 8 covering ≥95% of the train pool of the dataset with the longest reviews (Steam). Sequences are truncated from the end.
  * Width levels S/M/L give ≈50k / 200k / 500k non-embedding parameters, all within ±3% (`python scripts/count_params.py`).
  * Training: AdamW (weight decay 0.01), batch 64, ≤30 epochs, early stopping with patience 5 on validation macro-F1, best checkpoint restored, gradient clip 1.0, fp32.
* **Tuning:** exactly 10 predefined configs per method, selected by validation macro-F1 (ties go to the lower fit time). XGBoost uses early stopping during tuning, and its best round count is frozen.
* **Leakage:** validation drives tuning, early stopping, the per-family winners and the ablation method choice. Test data is evaluated only in final runs.
* **Crossover:** family winners at each size by validation, a paired per-seed test gap, threshold = the smallest size with mean gap ≥ 1.0, a test-selection sensitivity check, a Pareto check, and adjacent-size gains.
* **Ablation:**
  * Methods: the best TF-IDF and the best neural method by validation at n=8,000 in the main sweep, with frozen Kiwi-tuned hyperparameters.
  * Gap: computed with each family's validation-best tokenizer, alongside a check of whether the main conclusion still holds.

## 4. Cost measurement (per run, each in its own subprocess)

* **Fit time:** end-to-end (normalization, tokenization, vectoriser/vocabulary and training) and classifier-only.
* **Latency:** batch size 1 on 200 fixed test reviews after 20 warm-ups (median and p95), classifier-only and end-to-end (raw string → label). Measured on the **CPU for all methods** and also on the **accelerator** for neural methods, with `synchronize()` around every call.
* **Memory:** peak RSS (`ru_maxrss`, normalized for Linux KiB vs macOS bytes, plus a 10 ms psutil sampler), RSS just before fitting, and peak accelerator memory (CUDA `max_memory_allocated`; MPS sampled `driver_allocated_memory`).
* **Size:** serialized artifact size, plus parameter counts, tree node counts or non-zero coefficient counts.
* **Recorded per run:** `train_device`, `latency_devices`, `mamba_backend`, `precision`, `n_threads`, and a compact hardware block with a fingerprint.
* **Fairness checks:**
  * `aggregate.py` refuses results from more than one hardware fingerprint unless you pass `--allow-mixed-hardware`, in which case the tables are flagged.
  * `--parallel` is accepted only for tuning and smoke runs, and their timings are marked `timing_reportable: false`.

## 5. Deviations from the proposal

1. 5 seeds instead of 3, giving 675 instead of 405 final fits; confusion matrices are summed over 5 runs.
2. A new tokenizer ablation (120 additional fits).
3. Preprocessing is fully specified: the normalization rules and Kiwi `form/TAG` morphemes.
4. All reported numbers are mean ± sample std (ddof=1) over 5 seeds.

## 6. Layout

```
configs/      global.yaml, data.yaml, search_spaces/ (10 configs each), model_widths.yaml, frozen/
src/ksent/    hardware.py, data/ (download normalize clean split subsets stats), features/ (tokenize tfidf vocab),
              models/, train/, eval/ (metrics cost), pipeline.py, run_one.py, sweep.py, tune.py,
              analysis/ (aggregate crossover ablation make_tables make_figures make_numbers pilot_report check_results)
scripts/      00–08 *.sh, _common.sh, run_everything.sh, count_params.py
tests/        data integrity, preprocessing, models, hardware (CUDA/MPS-marked), analysis chain
demo/app.py   Gradio demo (normalized text, Kiwi tokens, nine predictions, confidence, live latency)
```
