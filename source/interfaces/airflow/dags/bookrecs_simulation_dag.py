from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.docker.operators.docker import DockerOperator
from dag_common import DEFAULT_ARGS, default_docker_args, docker_env

_FRACTIONS = [0.15, 0.30, 0.45, 0.60, 0.75]


def _stage_env(fraction: float) -> dict[str, str]:
    env = docker_env()
    env["BOOKRECS_SEED_BATCH_FRACTION"] = str(fraction)
    env["BOOKRECS_SEED_MAX_FRACTION"] = str(fraction)
    env["BOOKRECS_BATCH_RUN_NAME"] = f"simulation_stage_{int(fraction * 100):02d}pct"
    return env


with DAG(
    dag_id="bookrecs_simulation",
    default_args=DEFAULT_ARGS,
    description="Walk-forward simulation: train on 15/30/45/60/75% of dataset",
    schedule=None,
    start_date=datetime(2026, 4, 1),
    catchup=False,
    max_active_runs=1,
    tags=["bookrecs", "simulation", "ml"],
) as dag:
    prev_task = None

    for fraction in _FRACTIONS:
        pct = int(fraction * 100)
        stage_label = f"{pct:02d}pct"

        seed_task = DockerOperator(
            task_id=f"seed_{stage_label}",
            command="python -m source.interfaces.seed_interactions_entrypoint",
            environment={
                **docker_env(),
                "BOOKRECS_SEED_BATCH_FRACTION": str(fraction),
                "BOOKRECS_SEED_MAX_FRACTION": str(fraction),
            },
            **{k: v for k, v in default_docker_args().items() if k != "environment"},
        )

        train_task = DockerOperator(
            task_id=f"train_{stage_label}",
            command="python -m source.interfaces.simulation_batch_entrypoint",
            environment={
                **docker_env(),
                "BOOKRECS_BATCH_RUN_NAME": f"simulation_{stage_label}",
            },
            **{k: v for k, v in default_docker_args().items() if k != "environment"},
        )

        quality_task = TriggerDagRunOperator(
            task_id=f"quality_{stage_label}",
            trigger_dag_id="bookrecs_data_quality",
            wait_for_completion=True,
            poke_interval=30,
        )

        seed_task >> train_task >> quality_task

        if prev_task is not None:
            prev_task >> seed_task

        prev_task = quality_task
