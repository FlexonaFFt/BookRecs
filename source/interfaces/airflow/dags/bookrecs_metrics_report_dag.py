from __future__ import annotations

import json
import os
from datetime import datetime

import psycopg2 as psycopg
from airflow import DAG
from airflow.hooks.base import BaseHook
from airflow.operators.python import PythonOperator
from dag_common import DEFAULT_ARGS, PG_CONN_ID


def _pg_dsn() -> str:
    return BaseHook.get_connection(PG_CONN_ID).get_uri()


def report_metrics(**_) -> None:
    dsn = _pg_dsn()
    limit = int(os.getenv("BOOKRECS_REPORT_RUNS_LIMIT", "5"))

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT run_id, status, started_at, finished_at, metrics_json
                FROM pipeline_runs
                WHERE pipeline_name = 'bookrecs'
                ORDER BY finished_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            runs = cur.fetchall()

            cur.execute(
                "SELECT COUNT(*) FROM user_item_interactions WHERE event_type = 'seed'"
            )
            row = cur.fetchone()
            seeded_count = int(row[0]) if row else 0

            cur.execute(
                """
                SELECT interactions_count, trained_at
                FROM training_checkpoints
                ORDER BY trained_at DESC LIMIT 1
                """
            )
            checkpoint = cur.fetchone()

    print("[metrics-report] ════════════════════════════════", flush=True)
    print(f"[metrics-report] seeded interactions : {seeded_count}", flush=True)

    if checkpoint:
        last_count, last_trained = checkpoint
        growth = seeded_count - last_count
        print(f"[metrics-report] last training       : {last_trained}", flush=True)
        print(f"[metrics-report] interactions since  : +{growth}", flush=True)

    print(f"[metrics-report] last {limit} pipeline runs:", flush=True)

    for run_id, status, started_at, finished_at, metrics_json in runs:
        metrics = (
            json.loads(metrics_json)
            if isinstance(metrics_json, str)
            else (metrics_json or {})
        )
        ndcg = metrics.get("ndcg@10")
        recall = metrics.get("recall@10")
        duration = (
            int((finished_at - started_at).total_seconds())
            if finished_at and started_at
            else None
        )
        ndcg_str = f"{ndcg:.4f}" if ndcg is not None else "N/A"
        recall_str = f"{recall:.4f}" if recall is not None else "N/A"
        duration_str = f"{duration}s" if duration is not None else "N/A"
        print(
            f"[metrics-report]   {run_id}"
            f" | {status}"
            f" | ndcg@10={ndcg_str}"
            f" | recall@10={recall_str}"
            f" | duration={duration_str}",
            flush=True,
        )

    print("[metrics-report] ════════════════════════════════", flush=True)


with DAG(
    dag_id="bookrecs_metrics_report",
    default_args=DEFAULT_ARGS,
    description="Daily report on model metrics and interactions volume",
    schedule="0 6 * * *",
    start_date=datetime(2026, 4, 1),
    catchup=False,
    max_active_runs=1,
    tags=["bookrecs", "ml", "monitoring"],
) as dag:
    PythonOperator(
        task_id="print_metrics_report",
        python_callable=report_metrics,
    )
