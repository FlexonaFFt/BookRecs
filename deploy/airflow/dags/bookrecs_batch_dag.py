from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator


def _env_if_set(name: str) -> str | None:
    value = (os.getenv(name) or "").strip()
    return value or None


def _docker_env() -> dict[str, str]:
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
        "BOOKRECS_TRAIN_CANDIDATE_POOL_SIZE",
        "BOOKRECS_TRAIN_PER_SOURCE_LIMIT",
        "BOOKRECS_TRAIN_PRE_TOP_M",
        "BOOKRECS_TRAIN_FINAL_TOP_K",
        "BOOKRECS_TRAIN_CF_MODE",
        "BOOKRECS_TRAIN_CF_MAX_NEIGHBORS",
        "BOOKRECS_TRAIN_CF_MAX_ITEMS_PER_USER",
        "BOOKRECS_TRAIN_CONTENT_MAX_NEIGHBORS",
        "BOOKRECS_TRAIN_SEED",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]

    for key in passthrough_names:
        value = _env_if_set(key)
        if value is not None:
            env[key] = value

    return env


default_args = {
    "owner": "bookrecs",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="bookrecs_daily_batch",
    default_args=default_args,
    description="BookRecs daily batch pipeline",
    schedule="0 3 * * *",
    start_date=datetime(2026, 3, 1),
    catchup=True,
    max_active_runs=1,
    tags=["bookrecs", "batch", "ml"],
) as dag:
    run_batch = DockerOperator(
        task_id="run_batch_pipeline",
        image=_env_if_set("BOOKRECS_BATCH_IMAGE") or "bookrecs-pipeline:latest",
        api_version="auto",
        auto_remove="success",
        command="python -m source.interfaces.batch_entrypoint",
        docker_url=_env_if_set("DOCKER_HOST") or "unix://var/run/docker.sock",
        network_mode=_env_if_set("BOOKRECS_DOCKER_NETWORK") or "bridge",
        environment=_docker_env(),
        mount_tmp_dir=False,
    )

    promote_model = DockerOperator(
        task_id="promote_model",
        image=_env_if_set("BOOKRECS_BATCH_IMAGE") or "bookrecs-pipeline:latest",
        api_version="auto",
        auto_remove="success",
        command="python -m source.interfaces.promote_model_entrypoint",
        docker_url=_env_if_set("DOCKER_HOST") or "unix://var/run/docker.sock",
        network_mode=_env_if_set("BOOKRECS_DOCKER_NETWORK") or "bridge",
        environment={
            **_docker_env(),
            "BOOKRECS_PROMOTE_RUN_NAME": "batch_{{ ds_nodash }}",
            "BOOKRECS_ACTIVE_MODEL_POINTER": _env_if_set("BOOKRECS_ACTIVE_MODEL_POINTER")
            or "artifacts/runs/active_model.json",
            "BOOKRECS_PROMOTION_REQUIRE_SUCCESS": _env_if_set("BOOKRECS_PROMOTION_REQUIRE_SUCCESS") or "true",
            "BOOKRECS_PROMOTION_MIN_NDCG10": _env_if_set("BOOKRECS_PROMOTION_MIN_NDCG10") or "",
            "BOOKRECS_PROMOTION_MIN_RECALL10": _env_if_set("BOOKRECS_PROMOTION_MIN_RECALL10") or "",
            "BOOKRECS_PROMOTION_MIN_COLD_NDCG10": _env_if_set("BOOKRECS_PROMOTION_MIN_COLD_NDCG10") or "",
            "BOOKRECS_PROMOTION_MIN_COLD_RECALL10": _env_if_set("BOOKRECS_PROMOTION_MIN_COLD_RECALL10") or "",
        },
        mount_tmp_dir=False,
    )

    run_batch >> promote_model
