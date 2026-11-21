import csv

import pytest

from korean_sentiment.orchestration.executor import execute_plan


def test_one_seed_does_not_report_a_zero_sample_standard_deviation():
    from korean_sentiment.orchestration.aggregate import _mean_std

    assert _mean_std([0.7]) == (0.7, None)


def test_literal_na_review_text_survives_processed_csv_loading(isolated_paths):
    import pandas as pd
    from korean_sentiment.config import data_dir
    from korean_sentiment.data.prepare import load_split

    path = data_dir() / "processed/nsmc/train.csv"
    path.parent.mkdir(parents=True)
    texts = ["NA", "null", "nan", "None"]
    pd.DataFrame({"example_id": list("abcd"), "source_row_id": list("abcd"), "text": texts, "label": [0, 1, 0, 1]}).to_csv(path, index=False)
    assert load_split("nsmc", "train")["text"].tolist() == texts


def test_cpu_parallelism_cannot_launch_competing_neural_jobs(tmp_path):
    path = tmp_path / "plan.csv"
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "model_family"])
        writer.writeheader()
        writer.writerow({"run_id": "neural", "model_family": "neural"})
    with pytest.raises(ValueError, match="CPU-only"):
        execute_plan(path, parallelism=2)


def test_live_status_reports_pending_then_running(isolated_paths):
    from korean_sentiment.config import results_dir
    from korean_sentiment.orchestration.status import status_report
    from korean_sentiment.utils.io import atomic_write_csv, atomic_write_json

    job = {"run_id": "fixture", "phase": "part1_final", "dataset": "nsmc", "model": "tfidf_lr"}
    atomic_write_csv(results_dir() / "csv/experiment_plan.csv", [job], list(job))
    assert status_report()["overall"]["pending"] == 1
    atomic_write_json(results_dir() / "run_records/fixture/status.json", {"status": "running"})
    assert status_report()["overall"]["running"] == 1


def test_completed_marker_without_atomic_record_is_not_resumable(tmp_path):
    from korean_sentiment.orchestration.state import REQUIRED_COMPLETED, validate_completed
    from korean_sentiment.utils.io import atomic_write_json, atomic_write_text

    directory = tmp_path / "fixture"
    directory.mkdir()
    for filename in REQUIRED_COMPLETED:
        if filename != "SUCCESS":
            atomic_write_json(directory / filename, {})
    atomic_write_json(directory / "status.json", {"status": "completed"})
    atomic_write_json(directory / "record.json", {"run_id": "fixture", "status": "completed"})
    atomic_write_text(directory / "SUCCESS", "validated\n")
    assert validate_completed(directory)
    (directory / "record.json").unlink()
    assert not validate_completed(directory)


def test_failed_smoke_returns_nonzero(monkeypatch, capsys):
    from korean_sentiment import cli

    monkeypatch.setattr(cli, "run_smoke", lambda device: [{"status": "failed"}])
    with pytest.raises(SystemExit) as error:
        cli.main(["smoke"])
    assert error.value.code == 1


def test_unavailable_mamba_does_not_fail_local_smoke(monkeypatch):
    from korean_sentiment import cli

    monkeypatch.setattr(cli, "run_smoke", lambda device: [{"status": "blocked_on_mac"}])
    cli.main(["smoke"])


def test_archive_includes_operational_evidence_but_excludes_environment_exports(isolated_paths, monkeypatch):
    import tarfile
    from korean_sentiment.config import results_dir
    from korean_sentiment.orchestration.archive import archive_results

    monkeypatch.setattr("korean_sentiment.orchestration.archive.PROJECT_ROOT", isolated_paths)
    server = results_dir() / "server"
    server.mkdir()
    (server / "runtime_estimate.json").write_text("{}")
    (server / "pip_freeze.txt").write_text("not archived")
    scripts = isolated_paths / "scripts"
    scripts.mkdir()
    (scripts / "resume.sh").write_text("#!/bin/sh\n")
    run = results_dir() / "run_records/fixture"
    run.mkdir(parents=True)
    (run / "SUCCESS").write_text("validated\n")
    (run / "confusion_matrix.json").write_text("[[1, 0], [0, 1]]")
    (run / "run_attempt.log").write_text("preserved prior attempt traceback")
    (run / "model.pt").write_bytes(b"excluded checkpoint fixture")
    # Other archive roots belong to this repository, so isolate them as well.
    monkeypatch.setattr("korean_sentiment.orchestration.archive.CONFIG_DIR", isolated_paths / "configs")
    monkeypatch.setattr("korean_sentiment.orchestration.archive.logs_dir", lambda: isolated_paths / "logs")
    with tarfile.open(archive_results()) as archive:
        names = archive.getnames()
    assert any(name.endswith("server/runtime_estimate.json") for name in names)
    assert not any(name.endswith("pip_freeze.txt") for name in names)
    assert any(name.endswith("scripts/resume.sh") for name in names)
    assert any(name.endswith("fixture/SUCCESS") for name in names)
    assert any(name.endswith("fixture/confusion_matrix.json") for name in names)
    assert any(name.endswith("fixture/run_attempt.log") for name in names)
    assert not any(name.endswith("model.pt") for name in names)
