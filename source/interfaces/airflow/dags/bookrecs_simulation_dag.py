from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from dag_common import DEFAULT_ARGS, default_docker_args, docker_env, env_if_set

with DAG(
    dag_id="bookrecs_simulation",
    default_args=DEFAULT_ARGS,
    description="BookRecs temporal simulation — daily retraining on growing data slices",
    schedule="0 3 * * *",
    start_date=datetime(2026, 3, 1),
    catchup=True,
    max_active_runs=1,
    tags=["bookrecs", "simulation", "ml"],
) as dag:
    run_simulation = DockerOperator(
        task_id="run_simulation",
        command="python -m source.interfaces.simulation_batch_entrypoint",
        environment={
            **docker_env(),
            "BOOKRECS_SIMULATION_WINDOW_START": (
                env_if_set("BOOKRECS_SIMULATION_WINDOW_START") or "2026-03-01"
            ),
            "BOOKRECS_SIMULATION_WINDOW_END": (
                env_if_set("BOOKRECS_SIMULATION_WINDOW_END") or "2026-04-01"
            ),
            "BOOKRECS_SIMULATION_DATA_START": (
                env_if_set("BOOKRECS_SIMULATION_DATA_START") or "2011-01-01"
            ),
            "BOOKRECS_SIMULATION_DATA_END": (
                env_if_set("BOOKRECS_SIMULATION_DATA_END") or "2017-01-01"
            ),
        },
        **{k: v for k, v in default_docker_args().items() if k != "environment"},
    )

    promote_model = DockerOperator(
        task_id="promote_model",
        command="python -m source.interfaces.promote_model_entrypoint",
        environment={
            **docker_env(),
            "BOOKRECS_PROMOTE_RUN_NAME": "simulation_{{ ds_nodash }}",
            "BOOKRECS_ACTIVE_MODEL_POINTER": (
                env_if_set("BOOKRECS_ACTIVE_MODEL_POINTER")
                or "artifacts/runs/active_model.json"
            ),
            "BOOKRECS_PROMOTION_REQUIRE_SUCCESS": (
                env_if_set("BOOKRECS_PROMOTION_REQUIRE_SUCCESS") or "true"
            ),
            "BOOKRECS_PROMOTION_MIN_NDCG10": (
                env_if_set("BOOKRECS_PROMOTION_MIN_NDCG10") or ""
            ),
            "BOOKRECS_PROMOTION_MIN_RECALL10": (
                env_if_set("BOOKRECS_PROMOTION_MIN_RECALL10") or ""
            ),
            "BOOKRECS_PROMOTION_MIN_COLD_NDCG10": (
                env_if_set("BOOKRECS_PROMOTION_MIN_COLD_NDCG10") or ""
            ),
            "BOOKRECS_PROMOTION_MIN_COLD_RECALL10": (
                env_if_set("BOOKRECS_PROMOTION_MIN_COLD_RECALL10") or ""
            ),
        },
        **{k: v for k, v in default_docker_args().items() if k != "environment"},
    )

    run_simulation >> promote_model
