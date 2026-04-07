from __future__ import annotations

import os
from datetime import datetime

import psycopg2 as psycopg
from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.docker.operators.docker import DockerOperator
from dag_common import DEFAULT_ARGS, default_docker_args, docker_env


def _pg_dsn() -> str:
    dsn = (os.getenv("BOOKRECS_PG_DSN") or "").strip()
    if not dsn:
        raise ValueError("BOOKRECS_PG_DSN is required")
    return dsn


def check_volume_threshold(**context) -> bool:
    threshold = float(os.getenv("BOOKRECS_RETRAIN_THRESHOLD", "0.10"))
    dsn = _pg_dsn()

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM user_item_interactions WHERE event_type = 'seed'"
            )
            row = cur.fetchone()
            current_count = int(row[0]) if row else 0

            cur.execute(
                """
                SELECT interactions_count FROM training_checkpoints
                ORDER BY trained_at DESC LIMIT 1
                """
            )
            row = cur.fetchone()
            last_count = int(row[0]) if row else 0

    print(
        f"[retrain-check] current={current_count} last_trained={last_count}", flush=True
    )

    if last_count == 0 and current_count > 0:
        print(
            "[retrain-check] no previous training found, triggering first run",
            flush=True,
        )
        context["ti"].xcom_push(key="current_count", value=current_count)
        return True

    if current_count == 0:
        print("[retrain-check] no interactions yet, skipping", flush=True)
        return False

    if current_count < last_count:
        print(
            "[retrain-check] interactions count decreased since last checkpoint,"
            " triggering retrain",
            flush=True,
        )
        context["ti"].xcom_push(key="current_count", value=current_count)
        return True

    growth = (current_count - last_count) / last_count if last_count > 0 else 1.0
    print(f"[retrain-check] growth={growth:.1%} threshold={threshold:.1%}", flush=True)

    if growth >= threshold:
        print("[retrain-check] threshold reached, triggering retrain", flush=True)
        context["ti"].xcom_push(key="current_count", value=current_count)
        return True

    needed = int(last_count * (1 + threshold)) - current_count
    print(
        f"[retrain-check] threshold not reached, skipping"
        f" (need +{needed} more interactions to trigger)",
        flush=True,
    )
    return False


def save_checkpoint(**context) -> None:
    dsn = _pg_dsn()
    run_id = f"retrain_{context['ds_nodash']}"
    current_count = (
        context["ti"].xcom_pull(task_ids="check_volume", key="current_count") or 0
    )

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO training_checkpoints (run_id, interactions_count)"
                " VALUES (%s, %s)",
                (run_id, current_count),
            )
        conn.commit()

    print(
        f"[retrain-check] checkpoint saved run_id={run_id}"
        f" interactions={current_count}",
        flush=True,
    )


def _retrain_env() -> dict[str, str]:
    env = docker_env()
    env["BOOKRECS_SKIP_PREPARE"] = "true"
    env["BOOKRECS_SKIP_TRAIN"] = "false"
    env["BOOKRECS_BATCH_RUN_NAME"] = "retrain_{{ ds_nodash }}"
    return env


def _prepare_env() -> dict[str, str]:
    env = docker_env()
    env["BOOKRECS_SKIP_PREPARE"] = "false"
    env["BOOKRECS_SKIP_TRAIN"] = "true"
    env["BOOKRECS_TRAIN_RUN_NAME"] = "prepare_{{ ds_nodash }}"
    return env


with DAG(
    dag_id="bookrecs_retrain_on_volume",
    default_args=DEFAULT_ARGS,
    description="Trigger retraining when interactions grow by 10%",
    schedule="0 4 * * *",
    start_date=datetime(2026, 4, 1),
    catchup=False,
    max_active_runs=1,
    tags=["bookrecs", "ml", "retrain"],
) as dag:
    check_volume = ShortCircuitOperator(
        task_id="check_volume",
        python_callable=check_volume_threshold,
    )

    prepare_dataset = DockerOperator(
        task_id="prepare_dataset",
        command="python -m source.interfaces.pipeline_entrypoint",
        environment=_prepare_env(),
        **{k: v for k, v in default_docker_args().items() if k != "environment"},
    )

    train = DockerOperator(
        task_id="retrain",
        command="python -m source.interfaces.batch_entrypoint",
        environment=_retrain_env(),
        **{k: v for k, v in default_docker_args().items() if k != "environment"},
    )

    save_checkpoint_task = PythonOperator(
        task_id="save_checkpoint",
        python_callable=save_checkpoint,
    )

    trigger_data_quality = TriggerDagRunOperator(
        task_id="trigger_data_quality",
        trigger_dag_id="bookrecs_data_quality",
        wait_for_completion=True,
        poke_interval=30,
    )

    check_volume >> prepare_dataset >> train >> save_checkpoint_task >> trigger_data_quality
