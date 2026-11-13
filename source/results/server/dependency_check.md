# Verified ML dependencies

```json
{
  "utc": "2026-11-12T00:38:32.009349+00:00",
  "executable": "/home/dung/miniconda3/envs/ML/bin/python",
  "python": "3.11.18 (main, Oct 26 2026, 18:19:35) [GCC 14.3.0]",
  "cuda_available": true,
  "device": "NVIDIA GeForce RTX 4070",
  "cuda_tensor_sum": 16.0,
  "mamba_forward_backward": {
    "shape": [
      2,
      8,
      32
    ],
    "device": "cuda:0"
  },
  "packages": {
    "torch": "2.5.1+cu121",
    "scikit-learn": "1.9.1",
    "xgboost": "3.2.0",
    "mamba-ssm": "2.2.6.post3",
    "causal-conv1d": "1.6.0",
    "pytest": "9.2.0"
  }
}
```

Initial failure logs are preserved. GPU checks require host access outside the restricted execution sandbox.
