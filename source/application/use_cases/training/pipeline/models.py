from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainPipelineCommand:
    dataset_dir: str = "artifacts/tmp_preprocessed/goodreads_ya"
    output_root: str = "artifacts/runs"
    run_name: str | None = None
    eval_users_limit: int = 2000
    candidate_pool_size: int = 1000
    candidate_per_source_limit: int = 300
    pre_top_m: int = 300
    final_top_k: int = 10
    cf_max_neighbors: int = 120
    content_max_neighbors: int = 120
    seed: int = 42


@dataclass(frozen=True)
class TrainPipelineResult:
    run_id: str
    run_dir: str
    manifest_path: str
    metrics_path: str
    timings_path: str
    log_path: str
