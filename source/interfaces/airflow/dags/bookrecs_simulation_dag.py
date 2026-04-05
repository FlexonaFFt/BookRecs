from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.docker.operators.docker import DockerOperator
from dag_common import DEFAULT_ARGS, default_docker_args, docker_env

_FRACTIONS = [0.15, 0.30, 0.45, 0.60, 0.75]


def _train_env(fraction: float) -> dict[str, str]:
    pct = int(fraction * 100)
    stage_label = f"{pct:02d}pct"
    env = docker_env()
    env["BOOKRECS_BATCH_RUN_NAME"] = f"sim2_{stage_label}"
    env["BOOKRECS_TRAIN_DATA_FRACTION"] = str(fraction)
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

        train_task = DockerOperator(
            task_id=f"train_{stage_label}",
            command="python -m source.interfaces.simulation_batch_entrypoint",
            environment=_train_env(fraction),
            **{k: v for k, v in default_docker_args().items() if k != "environment"},
        )

        quality_task = TriggerDagRunOperator(
            task_id=f"quality_{stage_label}",
            trigger_dag_id="bookrecs_data_quality",
            wait_for_completion=True,
            poke_interval=30,
        )

        train_task >> quality_task

        if prev_task is not None:
            prev_task >> train_task

        prev_task = quality_task
