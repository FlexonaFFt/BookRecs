from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
# Описывает настройки приложения.
class Settings:
    pg_dsn: str
    s3_bucket: str
    s3_region: str
    s3_endpoint: str
    train_dataset_dir: str
    train_output_root: str
    train_eval_users_limit: int
    train_candidate_pool_size: int
    train_candidate_per_source_limit: int
    train_pre_top_m: int
    train_final_top_k: int
    train_cf_max_neighbors: int
    train_content_max_neighbors: int
    train_seed: int


def load_settings() -> Settings:
    def _int_env(name: str, default: int) -> int:
        raw = os.getenv(name, str(default)).strip()
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid int for {name}: {raw}") from exc
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")
        return value

    return Settings(
        pg_dsn=os.getenv("BOOKRECS_PG_DSN", ""),
        s3_bucket=os.getenv("BOOKRECS_S3_BUCKET", ""),
        s3_region=os.getenv("BOOKRECS_S3_REGION", "us-east-1"),
        s3_endpoint=os.getenv("BOOKRECS_S3_ENDPOINT", ""),
        train_dataset_dir=os.getenv("BOOKRECS_TRAIN_DATASET_DIR", "artifacts/tmp_preprocessed/goodreads_ya"),
        train_output_root=os.getenv("BOOKRECS_TRAIN_OUTPUT_ROOT", "artifacts/runs"),
        train_eval_users_limit=_int_env("BOOKRECS_TRAIN_EVAL_USERS_LIMIT", 2000),
        train_candidate_pool_size=_int_env("BOOKRECS_TRAIN_CANDIDATE_POOL_SIZE", 1000),
        train_candidate_per_source_limit=_int_env("BOOKRECS_TRAIN_PER_SOURCE_LIMIT", 300),
        train_pre_top_m=_int_env("BOOKRECS_TRAIN_PRE_TOP_M", 300),
        train_final_top_k=_int_env("BOOKRECS_TRAIN_FINAL_TOP_K", 10),
        train_cf_max_neighbors=_int_env("BOOKRECS_TRAIN_CF_MAX_NEIGHBORS", 120),
        train_content_max_neighbors=_int_env("BOOKRECS_TRAIN_CONTENT_MAX_NEIGHBORS", 120),
        train_seed=_int_env("BOOKRECS_TRAIN_SEED", 42),
    )
