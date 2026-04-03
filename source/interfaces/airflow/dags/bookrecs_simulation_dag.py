from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from dag_common import DEFAULT_ARGS, default_docker_args, docker_env, env_if_set

with DAG(
    dag_id="bookrecs_simulation",
    default_args=DEFAULT_ARGS,
    description="BookRecs temporal simulation — retraining on growing data slices",
    schedule="0 3 * * *",
    start_date=datetime(2026, 3, 1),
    catchup=True,
    max_active_runs=1,
    tags=["bookrecs", "simulation", "ml"],
) as dag:
    DockerOperator(
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
