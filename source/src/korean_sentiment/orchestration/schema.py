RUN_RECORD_FIELDS = [
    "run_id", "phase", "status", "dataset", "model", "model_family", "objective", "train_size",
    "train_fraction", "target_domain", "source_domains", "target_train_size", "source_size_per_domain",
    "seed", "config_id", "contrastive_lambda", "contrastive_temperature", "cross_domain_weight_alpha",
    "batch_sampler", "effective_batch_size", "device", "precision", "started_at", "finished_at",
    "duration_sec", "fit_time_sec", "best_epoch", "validation_macro_f1", "test_macro_f1", "test_accuracy",
    "test_macro_precision", "test_macro_recall", "precision_negative", "recall_negative", "f1_negative",
    "precision_positive", "recall_positive", "f1_positive", "tn", "fp", "fn", "tp",
    "classifier_latency_ms_mean", "classifier_latency_ms_p50", "classifier_latency_ms_p95",
    "classifier_latency_ms_std", "end_to_end_latency_ms_mean", "end_to_end_latency_ms_p50",
    "end_to_end_latency_ms_p95", "end_to_end_latency_ms_std", "latency_warmup_count",
    "latency_timed_repeat_count", "throughput_samples_per_sec", "peak_host_ram_mb", "peak_gpu_memory_mb",
    "serialized_size_mb", "total_parameters", "trainable_parameters", "tree_count", "node_count",
    "vocab_size", "feature_count", "max_sequence_length", "target_test_macro_f1", "source_1_test_macro_f1",
    "source_2_test_macro_f1", "mean_source_test_macro_f1", "all_domain_mean_macro_f1",
    "source_retention_difference", "same_sentiment_cross_domain_cosine",
    "different_sentiment_cross_domain_cosine", "cross_domain_sentiment_separation_gap",
    "same_domain_same_sentiment_cosine", "representation_pair_count", "representation_pair_seed",
    "deployed_parameter_count", "projection_parameters", "hardware_id", "hostname", "os", "python",
    "precision_mode", "cpu_threads", "git_commit", "dirty_worktree", "data_manifest_hash", "config_hash",
    "artifact_directory", "reused_run_ids", "error_type", "error_message",
]

PLAN_FIELDS = [
    "run_id", "phase", "dataset", "model", "model_family", "objective", "train_size", "train_fraction",
    "target_domain", "source_domains", "target_train_size", "source_size_per_domain", "seed", "config_id",
    "contrastive_lambda", "contrastive_temperature", "cross_domain_weight_alpha", "data_manifest_hash",
    "config_hash", "planning_state",
]

STATUSES = ("pending", "running", "completed", "failed", "interrupted", "blocked", "stale", "skipped")
