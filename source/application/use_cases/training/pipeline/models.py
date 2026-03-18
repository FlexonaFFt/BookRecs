from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
# Содержит входные данные команды пайплайна обучения.
class TrainPipelineCommand:
    dataset_dir: str = "artifacts/tmp_preprocessed/goodreads_ya"
    output_root: str = "artifacts/runs"
    run_name: str | None = None
    train_profile: str = "default"
    eval_users_limit: int = 2000
    cold_max_interactions: int = 5
    candidate_pool_size: int = 1000
    candidate_per_source_limit: int = 300
    pre_top_m: int = 300
    final_top_k: int = 10
    cf_mode: str = "auto"
    cf_max_neighbors: int = 120
    cf_max_items_per_user: int = 150
    content_max_neighbors: int = 120
    prerank_model: str = "auto"
    catboost_iterations: int = 250
    catboost_depth: int = 6
    catboost_learning_rate: float = 0.08
    seed: int = 42


@dataclass(frozen=True)
# Содержит результат выполнения пайплайна обучения.
class TrainPipelineResult:
    run_id: str
    run_dir: str
    manifest_path: str
    metrics_path: str
    timings_path: str
    log_path: str
