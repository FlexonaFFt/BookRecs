from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from dag_common import DEFAULT_ARGS, default_docker_args, docker_env

with DAG(
    dag_id="bookrecs_simulation",
    default_args=DEFAULT_ARGS,
    description="BookRecs simulation — daily retraining on existing dataset from S3",
    schedule="0 3 * * *",
    start_date=datetime(2026, 3, 1),
    catchup=True,
    max_active_runs=1,
    tags=["bookrecs", "simulation", "ml"],
) as dag:
    DockerOperator(
        task_id="run_simulation",
        command="python -m source.interfaces.simulation_batch_entrypoint",
        environment=docker_env(),
        **{k: v for k, v in default_docker_args().items() if k != "environment"},
    )
