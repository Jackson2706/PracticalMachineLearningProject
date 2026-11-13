import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("server_execution", Path(__file__).resolve().parents[2] / "scripts/server_execution.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


def measurements():
    return [{"dataset": "nsmc", "model": "transformer", "objective": "ce", "size": n,
             "duration_sec": 12.0, "validation_evaluation_sec": 1.0,
             "training": {"fit_time_sec": 10.0, "history": [{}, {}]}} for n in [200, 8000]]


def job(size):
    return {"dataset": "nsmc", "model": "transformer", "model_family": "neural", "objective": "ce",
            "train_size": size, "phase": "part1_final"}


def test_fixed_validation_cost_is_not_scaled_by_training_size():
    predictions = [server.predict_seconds(job(n), measurements(), "central") for n in [200, 1000, 8000]]
    assert predictions[0] == pytest.approx(predictions[1])
    assert predictions[1] == pytest.approx(predictions[2])


def test_forecast_cannot_substitute_another_dataset_stratum():
    request = {**job(8000), "dataset": "steam"}
    with pytest.raises(RuntimeError, match="missing pilot stratum"):
        server.predict_seconds(request, measurements(), "central")


def test_scenario_bounds_remain_ordered():
    values = [server.predict_seconds(job(1000), measurements(), case) for case in ["low", "central", "high"]]
    assert 0 < values[0] < values[1] < values[2]


def test_resource_summaries_preserve_unavailable_fields_and_match_conditions():
    records = [{"dataset": "nsmc", "model": "tfidf_lr", "train_size": "200", "device": "cpu", "fit_time_sec": value,
                "peak_gpu_memory_mb": None} for value in [2.0, 4.0]]
    rows = server.summarize_costs(records, ["dataset", "model", "train_size"])
    assert len(rows) == 1 and rows[0]["train_size"] == 200
    assert rows[0]["fit_time_sec_mean"] == 3.0
    assert rows[0]["peak_gpu_memory_mb_max"] is None
    assert rows[0]["classifier_latency_ms_p50_mean"] is None
