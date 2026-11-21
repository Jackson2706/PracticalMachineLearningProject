# Korean sentiment data-scale benchmark

This repository runs two linked experiments with Korean review data. Part 1 measures how nine classifiers change as the labeled training set grows. Part 2 asks whether 2,000 labeled examples from each of two existing domains reduce the number of new labels needed in a third target domain. All representations are learned from scratch.

## Study design

The datasets are NSMC movie reviews, Naver Shopping reviews, and Korean Steam reviews. NSMC uses its supplied binary label and official test set. Shopping ratings 1 and 2 map to negative, ratings 4 and 5 map to positive, and rating 3 is excluded if present. Steam uses the published binary recommendation field only as the target.

Preprocessing normalizes UTF-8 line endings, Unicode composition, and unnecessary whitespace. It keeps punctuation, emoji, emoticons, repeated characters, slang, and negation. Empty reviews are removed. Exact text-label duplicates are collapsed, while every occurrence of a normalized text with conflicting labels is removed. IDs, ratings, and recommendation fields never enter model features. The fixed split seed is 2026. NSMC retains its cleaned official test set and reserves 10% of the cleaned official training data for validation; cross-partition text duplicates are removed from the training side. Shopping and Steam use fixed stratified 80/10/10 splits.

Each dataset gets its own `N_full` after cleaning. Candidate sizes are 200, 500, 1,000, 2,000, 4,000, 8,000, 16,000, 32,000, 64,000, 128,000, and 256,000. Values at or above `N_full` are removed. The largest retained candidate is also removed when it is at least 90% of `N_full`, then the exact full size is appended. Seeds 13, 42, and 2026 create deterministic stratified orders, so every smaller set is a strict prefix and all models receive the same example IDs. Seed 777 is reserved for tuning.

Part 1 includes four word unigram/bigram TF-IDF pipelines with at most 10,000 L2-normalized features:

- logistic regression
- linear SVM
- random forest
- XGBoost

The five neural pipelines use the same deterministic Unicode-aware tokenizer and a vocabulary fitted only on the active training subset:

- average embeddings plus MLP
- embeddings plus LSTM
- embeddings plus GRU
- small Transformer encoder
- small Mamba classifier

Neural models use cross-entropy, AdamW, gradient clipping, validation macro-F1 early stopping, and best-checkpoint restoration. Mamba is an adapter to the maintained [`state-spaces/mamba`](https://github.com/state-spaces/mamba) package. It is intentionally blocked on macOS because the production implementation requires Linux, an NVIDIA GPU, CUDA PyTorch, and `mamba-ssm`. It is never replaced by an RNN or a local lookalike.

Each dataset-model pair has exactly ten predefined candidates. Tuning uses 8,000 examples, seed 777, and the fixed validation set. The selector maximizes validation macro-F1, then prefers lower validation inference latency and smaller serialized size. The winner is frozen across all sizes and final seeds. With the expected cleaned sizes, Part 1 has 270 tuning jobs and 837 final jobs, for 1,107 jobs. The planner derives the actual schedules and explains a different count instead of forcing these expectations.

## Contrastive bonus

Each bonus rotation holds one target domain at 200, 500, 1,000, or 2,000 labels and adds 2,000 training examples from each other domain. Those source and target examples come from the Part 1 nested manifests. Validation and test examples remain untouched. Source labels are treated as previously available data, so reported savings refer only to new target-domain labels; they are not total annotation savings.

Every comparison contains these five methods:

- the best TF-IDF Part 1 result at the same target size
- the target-only Part 1 Transformer with cross-entropy
- pooled multi-domain Transformer with cross-entropy
- pooled Transformer with standard supervised contrastive learning
- the proposed domain-balanced cross-domain supervised contrastive method, DB-CDSupCon

The first two are references to existing Part 1 run IDs. They are not retrained or selected using bonus results. The three new methods share the target's frozen Transformer settings, pooled examples, vocabulary rules, six-cell domain-label-balanced sampler, optimizer rules, budget, and seed. A deployed classifier uses only the masked-mean Transformer representation and binary head. SupCon variants add a two-layer `Linear -> ReLU -> Linear -> L2 normalization` projection head during training, then bypass it during inference.

Standard SupCon tunes four combinations from `lambda in {0.1, 0.5}` and `tau in {0.07, 0.10}`. DB-CDSupCon tunes eight combinations by adding `alpha in {2.0, 4.0}` for same-label cross-domain positives. Positive weights are normalized per anchor. The tuning target is target-domain validation macro-F1, followed by mean source validation macro-F1 and training time. This produces 36 bonus tuning jobs. The final grid has 3 new objectives x 3 targets x 4 sizes x 3 seeds = 108 jobs. Part 2 therefore adds 144 jobs, and the complete expected project count is 1,251.

## Environment

Python 3.11 and the Conda environment name `ML` are mandatory on both machines. The setup does not create a venv, Poetry environment, or a second Conda environment.

Begin an interactive setup with:

```bash
conda create -n ML python=3.11 -y
conda activate ML
```

The setup scripts are idempotent, so the explicit `conda create` command is only needed when `ML` does not exist. Noninteractive scripts locate Conda and use `conda run -n ML`; they do not assume activation survives another shell invocation. After setup, they save a from-history Conda export, an explicit package list, a pip freeze, and a hardware manifest under `source/results/artifacts/`.

Supported overrides are `KS_DEVICE`, `KS_DATA_DIR`, `KS_RESULTS_DIR`, `KS_NUM_WORKERS`, `KS_CPU_THREADS`, `KS_MAX_PARALLEL_CPU_JOBS`, `KS_MAX_PARALLEL_GPU_JOBS`, and `KS_PRECISION`. Copy `.env.example` values into your shell or job script. The code does not read or store credentials.

### Apple-silicon Mac

```bash
conda create -n ML python=3.11 -y        # only if ML is absent
conda activate ML
./run_all.sh setup-mac
./run_all.sh download
./run_all.sh preprocess
./run_all.sh test
./run_all.sh smoke
./run_all.sh plan
```

The Mac setup supports CPU and MPS. The smoke command records Mamba as `blocked_on_mac` and still verifies the other eight pipelines plus pooled CE, SupCon, and DB-CDSupCon. Do not run a production grid on the Mac. Inspect `source/results/csv/experiment_plan.csv`, then commit or copy source and configuration files only. Generated data, environments, checkpoints, logs, and results are ignored.

### Linux/NVIDIA cloud

The cloud script requires Linux, `nvidia-smi`, and a working driver. It installs CUDA-capable PyTorch first, then installs `mamba-ssm` with the upstream `MAMBA_KEEP_CUDA_BUILD=TRUE` and `--no-build-isolation` requirements. If the default PyTorch wheel does not match the rented machine, set `KS_TORCH_INDEX_URL` to the index URL selected from the official [PyTorch installer](https://pytorch.org/get-started/locally/). The preflight refuses neural full runs until a real Mamba CUDA forward and backward pass succeeds.

```bash
conda create -n ML python=3.11 -y        # only if ML is absent
conda activate ML
./run_all.sh setup-cloud
./run_all.sh preflight --cloud-neural
./run_all.sh download
./run_all.sh preprocess
./run_all.sh test
./run_all.sh smoke --device cuda
./run_all.sh pilot --model transformer --max-jobs 1
./run_all.sh plan
```

Inspect the pilot's measured time and resource record before starting the grid. The code does not invent an ETA.

Use `tmux`:

```bash
tmux new -s korean-sentiment
./run_all.sh benchmark-all --noninteractive-confirm-expensive
```

Or use `nohup`:

```bash
nohup ./run_all.sh benchmark-all --noninteractive-confirm-expensive > benchmark-cloud.log 2>&1 &
```

After Part 1 finishes and aggregates:

```bash
./run_all.sh aggregate
./run_all.sh bonus-plan
./run_all.sh bonus-all --noninteractive-confirm-expensive
./run_all.sh archive-results
```

`bonus-plan` writes 36 tuning rows, 108 new final rows, and a separate table of the two reused Part 1 reference types for every target-size-seed condition.

## Commands

`run_all.sh` accepts:

```text
setup-mac             create or validate ML and verify CPU/MPS
setup-cloud           create or validate ML and verify CUDA/Mamba
preflight             inspect software, hardware, paths, disk, and data readiness
download              download and validate all public raw files
preprocess            clean, split, create nested subsets, and validate manifests
test                  run fixture-only unit and integration tests
smoke                 exercise every locally available pipeline and both contrastive losses
pilot                 run a small measured representative job
plan                  write Part 1 tuning and final plans without training
tune                  run Part 1 tuning and freeze selected configurations
full                  run the resumable Part 1 final plan
aggregate             rebuild Part 1 and shared CSV outputs
bonus-plan            write bonus plans and Part 1 reference links
bonus-tune            run contrastive tuning and freeze its parameters
bonus-full            run the 108 new bonus jobs
bonus-aggregate       rebuild bonus CSV outputs
benchmark-all         execute the complete Part 1 workflow
bonus-all             execute Part 2 after its Part 1 dependencies
both                  run Part 1 and Part 2 in order
status                report planned and observed states
archive-results       archive small results without raw data or large checkpoints
all                   run both parts, then archive
```

`tune`, `full`, `bonus-tune`, `bonus-full`, `benchmark-all`, `bonus-all`, `both`, and `all` require `--confirm-expensive`. Use `--noninteractive-confirm-expensive` only when intentionally starting a detached cloud job.

Filters pass to the executor. Examples:

```bash
./run_all.sh full --confirm-expensive --dataset nsmc --model tfidf_lr --training-size 200 --seed 13
./run_all.sh bonus-full --confirm-expensive --target-domain steam --objective db_cdsupcon --target-training-size 500
./run_all.sh full --confirm-expensive --run-id part1-0123456789abcdefabcd
./run_all.sh tune --confirm-expensive --max-jobs 10 --cpu-parallelism 2
```

One neural process per GPU is enforced by default. Increase concurrency only after a successful pilot shows enough VRAM. CPU thread and worker limits remain explicit in every record.

## Resumption and failures

Each run owns `source/results/run_records/<run_id>/`. The run ID includes phase, data role, model, objective, sizes, seed, configuration ID, data-manifest hash, and configuration hash. A lock prevents duplicate execution. A dead local PID or expired remote lock is marked interrupted and recovered. A completed run is skipped only when all required files and the final `SUCCESS` marker validate. Changed data or configuration hashes mark an old run stale. Partial directories are never treated as completed, and normal resumption never deletes datasets, checkpoints, logs, or results.

The full grid continues after an individual failure unless `--fail-fast` is supplied. Inspect failures with:

```bash
./run_all.sh status
find source/results/run_records -name error.json -print
```

Resume with the same command. Add filters to narrow the retry. `--force` reruns an otherwise valid completed job without deleting its existing directory.

Run directories contain resolved configuration, metadata, status, a timestamped log, metrics, resources, confusion matrix, optional model and embedding artifacts, and an error traceback when applicable. The success marker appears only after required artifacts pass validation.

## Data

The repository ships the split and subset manifests (`source/data/splits/*.csv`, `source/data/manifests/`), dataset statistics, configurations, and the aggregated result CSVs. Raw downloads, processed review files, checkpoints, model artifacts, and embeddings are not committed. Recreate the data with:

```bash
./run_all.sh download     # fetch and validate the raw files into source/data/raw/
./run_all.sh preprocess   # rebuild source/data/processed/ plus the split and subset manifests
```

Preprocessing is deterministic, so the regenerated manifests match the committed ones.

## Data and result locations

- `source/data/raw/`: verified immutable downloads
- `source/data/processed/`: model-ready fixed splits
- `source/data/splits/`: immutable split manifests
- `source/data/manifests/`: download, cleaning, subset, and content-hash manifests
- `source/results/plans/`: tuning, final, bonus, and reference plans
- `source/results/run_records/`: atomic per-run records and failures
- `source/results/artifacts/`: retained models and environment/hardware snapshots
- `source/results/embeddings/`: optional deterministic representation samples
- `source/results/csv/`: rebuilt master tables
- `source/results/archives/`: timestamped result packages
- `source/logs/`: shell and run logs

Parallel workers never append to a shared CSV. Rebuild tables at any time with:

```bash
./run_all.sh aggregate
./run_all.sh bonus-aggregate
```

The CSV directory includes dataset statistics, plans, tuning trials, frozen selections, run-level metrics, three-seed summaries, crossovers, resources, statuses, bonus label efficiency, and representation diagnostics. A condition with fewer than three completed seeds is marked incomplete. Raw numerical precision is retained.

Monitor storage with portable commands such as `du -sh source/data source/results source/checkpoints`. `archive-results` includes configurations, manifests, CSVs, logs, and small run metadata. It excludes raw data, retained model artifacts, embeddings, and checkpoints.

Copy code to a cloud host with placeholders:

```bash
rsync -av --exclude 'source/data/raw/' --exclude 'source/results/' ./ USER@HOST:/PATH/PROJECT/
scp -r ./source ./scripts ./run_all.sh USER@HOST:/PATH/PROJECT/
```

Copy the result archive back:

```bash
rsync -av USER@HOST:/PATH/PROJECT/source/results/archives/ ./source/results/archives/
scp USER@HOST:/PATH/PROJECT/source/results/archives/korean_sentiment_results_TIMESTAMP.tar.gz ./
```

If a dataset host is temporarily unavailable, download the exact file URL from `source/configs/datasets.yaml` in a browser or with another trusted machine, copy it to `source/data/raw/<dataset>/<filename>`, and rerun `./run_all.sh download`. The downloader validates the schema and records the observed SHA-256; the repository does not contain an unverified hard-coded checksum.

## Interpretation and limitations

The crossover output asks when the best complete neural mean exceeds the best complete TF-IDF mean by at least 0.01 macro-F1. It records the first observed crossover and the first sustained crossover, defined as satisfying the criterion at that size and every larger tested size. A missing or unobserved threshold stays explicit. Three seeds describe variation but do not establish statistical significance.

Ratings and recommendation flags are noisy sentiment labels. The three domains differ in vocabulary, review style, and label mechanism. Learned-from-scratch token representations may need more data than pretrained language models, which are outside this study. TF-IDF and neural timing fields name whether preprocessing is included; they should not be treated as the same measurement without that distinction. Mamba adds Linux/CUDA build constraints. The complete 1,251-job protocol is computationally expensive.
