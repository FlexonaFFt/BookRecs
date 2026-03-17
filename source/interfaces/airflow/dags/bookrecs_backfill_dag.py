from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.providers.docker.operators.docker import DockerOperator

from _common import DEFAULT_ARGS, default_docker_args, docker_env


with DAG(
    dag_id="bookrecs_backfill",
    default_args=DEFAULT_ARGS,
    description="BookRecs manual backfill / offline replay",
    schedule=None,
    start_date=datetime(2026, 3, 1),
    catchup=False,
    max_active_runs=1,
    params={
        "end_date": Param("2026-03-10", type="string"),
        "days": Param(5, type="integer", minimum=1),
        "promote": Param(False, type="boolean"),
    },
    tags=["bookrecs", "batch", "backfill", "ml"],
) as dag:
    run_backfill = DockerOperator(
        task_id="run_backfill",
        command="python -m source.interfaces.batch_backfill_entrypoint",
        environment={
            **docker_env(),
            "BOOKRECS_BATCH_END_DATE": "{{ params.end_date }}",
            "BOOKRECS_BATCH_BACKFILL_DAYS": "{{ params.days }}",
            "BOOKRECS_BATCH_BACKFILL_PROMOTE": "{{ params.promote }}",
        },
        **{
            key: value
            for key, value in default_docker_args().items()
            if key != "environment"
        },
    )
