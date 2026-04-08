from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from dag_common import DEFAULT_ARGS, default_docker_args, docker_env, docker_secret_env

with DAG(
    dag_id="bookrecs_seed_interactions",
    default_args=DEFAULT_ARGS,
    description="Seed next batch of interactions from dataset into postgres",
    schedule="0 2 * * *",
    start_date=datetime(2026, 4, 1),
    catchup=False,
    max_active_runs=1,
    tags=["bookrecs", "simulation", "data"],
) as dag:
    DockerOperator(
        task_id="seed_interactions",
        command="python -m source.interfaces.seed_interactions_entrypoint",
        environment={**docker_env(), **docker_secret_env()},
        **{k: v for k, v in default_docker_args().items() if k != "environment"},
    )
