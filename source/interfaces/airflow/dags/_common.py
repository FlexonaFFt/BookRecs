from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

from docker.types import Mount


def env_if_set(name: str) -> str | None:
    value = (os.getenv(name) or "").strip()
    return value or None


def docker_env() -> dict[str, str]:
    env: dict[str, str] = {
        "BOOKRECS_BATCH_EXECUTION_DATE": "{{ ds }}",
        "BOOKRECS_BATCH_RUN_NAME": "",
    }

    passthrough_names = [
        "BOOKRECS_DATASET_NAME",
        "BOOKRECS_RAW_DIR",
        "BOOKRECS_BOOKS_RAW_URI",
        "BOOKRECS_INTERACTIONS_RAW_URI",
        "BOOKRECS_STORE_BACKEND",
        "BOOKRECS_REGISTRY_BACKEND",
        "BOOKRECS_S3_PREFIX",
        "BOOKRECS_S3_BUCKET",
        "BOOKRECS_S3_REGION",
        "BOOKRECS_S3_ENDPOINT",
        "BOOKRECS_PG_DSN",
        "BOOKRECS_PG_MIGRATION_PATH",
        "BOOKRECS_SKIP_PREPARE",
        "BOOKRECS_SKIP_TRAIN",
        "BOOKRECS_RUN_MIGRATE",
        "BOOKRECS_TRAIN_OUTPUT_ROOT",
        "BOOKRECS_TRAIN_DATASET_DIR",
        "BOOKRECS_TRAIN_EVAL_USERS_LIMIT",
        "BOOKRECS_COLD_MAX_INTERACTIONS",
        "BOOKRECS_TRAIN_CANDIDATE_POOL_SIZE",
        "BOOKRECS_TRAIN_PER_SOURCE_LIMIT",
        "BOOKRECS_TRAIN_PRE_TOP_M",
        "BOOKRECS_TRAIN_FINAL_TOP_K",
        "BOOKRECS_TRAIN_CF_MODE",
        "BOOKRECS_TRAIN_CF_MAX_NEIGHBORS",
        "BOOKRECS_TRAIN_CF_MAX_ITEMS_PER_USER",
        "BOOKRECS_TRAIN_CONTENT_MAX_NEIGHBORS",
        "BOOKRECS_TRAIN_PRERANK_MODEL",
        "BOOKRECS_TRAIN_CATBOOST_ITERATIONS",
        "BOOKRECS_TRAIN_CATBOOST_DEPTH",
        "BOOKRECS_TRAIN_CATBOOST_LEARNING_RATE",
        "BOOKRECS_TRAIN_SEED",
        "BOOKRECS_TRAIN_STDOUT_FORMAT",
        "BOOKRECS_TRAIN_PROFILE",
        "BOOKRECS_TRAIN_AUTO_TUNE",
        "BOOKRECS_ACTIVE_MODEL_POINTER",
        "BOOKRECS_PROMOTION_REQUIRE_SUCCESS",
        "BOOKRECS_PROMOTION_MIN_NDCG10",
        "BOOKRECS_PROMOTION_MIN_RECALL10",
        "BOOKRECS_PROMOTION_MIN_COLD_NDCG10",
        "BOOKRECS_PROMOTION_MIN_COLD_RECALL10",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]

    for key in passthrough_names:
        value = env_if_set(key)
        if value is not None:
            env[key] = value

    return env


def docker_mounts() -> list[Mount]:
    host_project_dir = env_if_set("BOOKRECS_HOST_PROJECT_DIR")
    if host_project_dir is None:
        return []

    project_dir = Path(host_project_dir).expanduser()
    mounts = [
        Mount(source=str(project_dir / "data"), target="/app/data", type="bind"),
        Mount(source=str(project_dir / "artifacts"), target="/app/artifacts", type="bind"),
    ]
    return mounts


def default_docker_args() -> dict[str, object]:
    return {
        "image": env_if_set("BOOKRECS_BATCH_IMAGE") or "bookrecs-pipeline:latest",
        "api_version": "auto",
        "auto_remove": "success",
        "docker_url": env_if_set("DOCKER_HOST") or "unix://var/run/docker.sock",
        "network_mode": env_if_set("BOOKRECS_DOCKER_NETWORK") or "bookrecs_net",
        "environment": docker_env(),
        "mount_tmp_dir": False,
        "mounts": docker_mounts(),
    }


DEFAULT_ARGS = {
    "owner": "bookrecs",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}
