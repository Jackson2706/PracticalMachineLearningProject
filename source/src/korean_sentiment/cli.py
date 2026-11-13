from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from korean_sentiment.config import results_dir
from korean_sentiment.data.download import download_all
from korean_sentiment.data.prepare import preprocess_all
from korean_sentiment.orchestration.aggregate import aggregate_all
from korean_sentiment.orchestration.archive import archive_results
from korean_sentiment.orchestration.executor import execute_plan, select_bonus_configs, select_part1_configs
from korean_sentiment.orchestration.planner import build_bonus_plans, build_part1_plans, validate_bonus_dependencies
from korean_sentiment.orchestration.preflight import preflight
from korean_sentiment.orchestration.smoke import run_smoke
from korean_sentiment.orchestration.status import status_report


def _filters(arguments: argparse.Namespace) -> dict:
    return {
        "dataset": arguments.dataset,
        "target_domain": arguments.target_domain,
        "model": arguments.model,
        "objective": arguments.objective,
        "size": arguments.training_size,
        "target_size": arguments.target_training_size,
        "seed": arguments.seed,
        "run_id": arguments.run_id,
    }


def _execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset")
    parser.add_argument("--target-domain")
    parser.add_argument("--model")
    parser.add_argument("--objective")
    parser.add_argument("--training-size", type=int)
    parser.add_argument("--target-training-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--run-id")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--cpu-parallelism", type=int)
    parser.add_argument("--gpu-parallelism", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")


def _execute(arguments: argparse.Namespace, plan_name: str) -> list[dict]:
    if arguments.device:
        os.environ["KS_DEVICE"] = arguments.device
    cpu_parallelism = arguments.cpu_parallelism or int(os.environ.get("KS_MAX_PARALLEL_CPU_JOBS", "1"))
    gpu_parallelism = arguments.gpu_parallelism or int(os.environ.get("KS_MAX_PARALLEL_GPU_JOBS", "1"))
    if gpu_parallelism > 1:
        raise RuntimeError("one neural process per GPU is the default; explicit multi-GPU scheduling is not implemented")
    return execute_plan(
        results_dir() / "plans" / plan_name,
        _filters(arguments),
        arguments.max_jobs,
        max(cpu_parallelism, gpu_parallelism),
        arguments.force,
        arguments.fail_fast,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ks-benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download").add_argument("--force", action="store_true")
    subparsers.add_parser("preprocess")
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--cloud-neural", action="store_true")
    preflight_parser.add_argument("--require-data", action="store_true")
    subparsers.add_parser("plan")
    subparsers.add_parser("bonus-plan")
    subparsers.add_parser("aggregate")
    subparsers.add_parser("bonus-aggregate")
    subparsers.add_parser("status")
    subparsers.add_parser("archive-results")
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--device")
    for command in ("tune", "full", "bonus-tune", "bonus-full", "pilot"):
        _execution_arguments(subparsers.add_parser(command))
    arguments = parser.parse_args(argv)
    if arguments.command == "download":
        result = download_all(arguments.force)
    elif arguments.command == "preprocess":
        result = preprocess_all()
    elif arguments.command == "preflight":
        result = preflight(arguments.cloud_neural, arguments.require_data)
    elif arguments.command == "plan":
        result = build_part1_plans()
    elif arguments.command == "bonus-plan":
        validate_bonus_dependencies()
        result = build_bonus_plans()
    elif arguments.command == "tune":
        result = _execute(arguments, "part1_tuning_plan.csv")
        if not any(item["status"] in {"failed", "blocked"} for item in result):
            select_part1_configs()
            build_part1_plans()
    elif arguments.command == "full":
        from korean_sentiment.config import TFIDF_MODELS

        preflight(require_cloud_neural=arguments.model not in TFIDF_MODELS, require_data=True)
        result = _execute(arguments, "part1_final_plan.csv")
    elif arguments.command == "bonus-tune":
        result = _execute(arguments, "bonus_tuning_plan.csv")
        if not any(item["status"] in {"failed", "blocked"} for item in result):
            select_bonus_configs()
            build_bonus_plans()
    elif arguments.command == "bonus-full":
        preflight(require_cloud_neural=True, require_data=True)
        result = _execute(arguments, "bonus_final_plan.csv")
    elif arguments.command == "pilot":
        arguments.max_jobs = arguments.max_jobs or 1
        result = _execute(arguments, "part1_tuning_plan.csv")
    elif arguments.command in {"aggregate", "bonus-aggregate"}:
        result = aggregate_all()
    elif arguments.command == "status":
        result = status_report()
    elif arguments.command == "archive-results":
        result = {"archive": str(archive_results())}
    elif arguments.command == "smoke":
        result = run_smoke(arguments.device)
    else:
        raise AssertionError(arguments.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if arguments.command in {"tune", "full", "bonus-tune", "bonus-full", "pilot", "smoke"}:
        if any(item.get("status") in {"failed", "blocked", "blocked_on_mac"} for item in result):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
