"""ksent: training-data thresholds for cost-effective Korean sentiment analysis."""

# Import xgboost before torch. On macOS the real fix is a single OpenMP runtime
# (scripts/00_setup_env.sh links xgboost to torch's libomp); a fixed import order
# additionally keeps library initialisation deterministic across entry points.
try:  # pragma: no cover - import side effect only
    import xgboost as _xgboost  # noqa: F401
except Exception:
    pass

__version__ = "0.1.0"
