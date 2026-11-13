# Numerical reproducibility notes

Runs fix Python/NumPy/PyTorch seeds and set CUBLAS_WORKSPACE_CONFIG=:4096:8. PyTorch deterministic-algorithm mode uses warn_only=True. The repository pilot logged a warning that memory-efficient attention defaults to a non-deterministic algorithm (repository_pilot_20261112T003912Z.log). Results are therefore not claimed to be bitwise deterministic. The three final seeds describe variation and do not establish statistical significance.

Training uses CUDA BF16 autocast with FP32 model parameters and optimizer state. Validation and deployed prediction use the pipeline's default FP32 forward path outside autocast. Reported precision therefore describes training autocast, not a claim that every inference operation is BF16.

Classifier latency starts from prepared features. End-to-end inference latency includes tokenization/encoding plus prediction for processed input text; one-time dataset download/cleaning/splitting is measured separately. The configured latency procedure uses 10 warm-ups and 50 timed calls; throughput uses batch size 128. Device and CPU-thread fields must be consulted when comparing CPU TF-IDF and CUDA neural costs.
