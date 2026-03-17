from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

from dag_common import DEFAULT_ARGS, default_docker_args, docker_env, env_if_set

with DAG(
    dag_id="bookrecs_daily_batch",
    default_args=DEFAULT_ARGS,
    description="BookRecs daily batch pipeline",
    schedule="0 3 * * *",
    start_date=datetime(2026, 3, 1),
    catchup=True,
    max_active_runs=1,
    tags=["bookrecs", "batch", "ml"],
) as dag:
    run_batch = DockerOperator(
        task_id="run_batch_pipeline",
        command="python -m source.interfaces.batch_entrypoint",
        **default_docker_args(),
    )

    promote_model = DockerOperator(
        task_id="promote_model",
        command="python -m source.interfaces.promote_model_entrypoint",
        environment={
            **docker_env(),
            "BOOKRECS_PROMOTE_RUN_NAME": "batch_{{ ds_nodash }}",
            "BOOKRECS_ACTIVE_MODEL_POINTER": env_if_set("BOOKRECS_ACTIVE_MODEL_POINTER")
            or "artifacts/runs/active_model.json",
            "BOOKRECS_PROMOTION_REQUIRE_SUCCESS": env_if_set("BOOKRECS_PROMOTION_REQUIRE_SUCCESS") or "true",
            "BOOKRECS_PROMOTION_MIN_NDCG10": env_if_set("BOOKRECS_PROMOTION_MIN_NDCG10") or "",
            "BOOKRECS_PROMOTION_MIN_RECALL10": env_if_set("BOOKRECS_PROMOTION_MIN_RECALL10") or "",
            "BOOKRECS_PROMOTION_MIN_COLD_NDCG10": env_if_set("BOOKRECS_PROMOTION_MIN_COLD_NDCG10") or "",
            "BOOKRECS_PROMOTION_MIN_COLD_RECALL10": env_if_set("BOOKRECS_PROMOTION_MIN_COLD_RECALL10") or "",
        },
        **{
            key: value
            for key, value in default_docker_args().items()
            if key != "environment"
        },
    )

    run_batch >> promote_model
