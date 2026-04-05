from __future__ import annotations

import json
import os
from datetime import datetime

import psycopg
from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from dag_common import DEFAULT_ARGS, default_docker_args, docker_env

_PROMOTE_TASK = "promote_model"
_SKIP_TASK = "skip_promotion"


def _pg_dsn() -> str:
    dsn = (os.getenv("BOOKRECS_PG_DSN") or "").strip()
    if not dsn:
        raise ValueError("BOOKRECS_PG_DSN is required")
    return dsn


def _get_float(d: dict, key: str) -> float | None:
    val = d.get(key)
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def decide_promotion(**context) -> str:
    dsn = _pg_dsn()
    min_ndcg = float(os.getenv("BOOKRECS_QUALITY_MIN_NDCG10", "0.0"))
    min_recall = float(os.getenv("BOOKRECS_QUALITY_MIN_RECALL10", "0.0"))
    force_after_skips = int(os.getenv("BOOKRECS_QUALITY_FORCE_AFTER_SKIPS", "2"))

    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            """
            SELECT run_id, status, metrics_json
            FROM pipeline_runs
            WHERE pipeline_name = 'bookrecs'
              AND status = 'SUCCESS'
            ORDER BY finished_at DESC
            LIMIT 3
            """
        ).fetchall()

    if not rows:
        print("[data-quality] no successful runs found, skipping", flush=True)
        return _SKIP_TASK

    latest_run_id, _, metrics_json = rows[0]
    metrics = (
        json.loads(metrics_json)
        if isinstance(metrics_json, str)
        else (metrics_json or {})
    )

    ndcg = _get_float(metrics, "ndcg_at_10")
    recall = _get_float(metrics, "recall_at_10")

    print(
        f"[data-quality] latest run={latest_run_id}"
        f" ndcg@10={ndcg} recall@10={recall}",
        flush=True,
    )

    meets_threshold = (ndcg is None or ndcg >= min_ndcg) and (
        recall is None or recall >= min_recall
    )

    consecutive_skips = 0
    for _, _, m in rows[1:]:
        m_dict = json.loads(m) if isinstance(m, str) else (m or {})
        if not m_dict.get("promoted", False):
            consecutive_skips += 1

    force_promote = consecutive_skips >= force_after_skips

    if meets_threshold:
        print("[data-quality] metrics OK → promote", flush=True)
        context["ti"].xcom_push(key="run_id", value=latest_run_id)
        return _PROMOTE_TASK

    if force_promote:
        print(
            f"[data-quality] skipped {consecutive_skips}x in a row → force promote",
            flush=True,
        )
        context["ti"].xcom_push(key="run_id", value=latest_run_id)
        return _PROMOTE_TASK

    print(
        f"[data-quality] metrics below threshold"
        f" (ndcg={ndcg} < {min_ndcg} or recall={recall} < {min_recall}),"
        " skipping",
        flush=True,
    )
    return _SKIP_TASK


def skip_promotion(**_) -> None:
    print("[data-quality] promotion skipped", flush=True)


def _promote_env() -> dict[str, str]:
    env = docker_env()
    env["BOOKRECS_PROMOTE_RUN_NAME"] = (
        "{{ ti.xcom_pull(task_ids='decide_promotion', key='run_id') }}"
    )
    return env


with DAG(
    dag_id="bookrecs_data_quality",
    default_args=DEFAULT_ARGS,
    description="Decide whether to promote the latest trained model to production",
    schedule=None,
    start_date=datetime(2026, 4, 1),
    catchup=False,
    max_active_runs=1,
    tags=["bookrecs", "ml", "quality"],
) as dag:
    decide = BranchPythonOperator(
        task_id="decide_promotion",
        python_callable=decide_promotion,
    )

    promote = DockerOperator(
        task_id=_PROMOTE_TASK,
        command="python -m source.interfaces.promote_model_entrypoint",
        environment=_promote_env(),
        **{k: v for k, v in default_docker_args().items() if k != "environment"},
    )

    skip = PythonOperator(
        task_id=_SKIP_TASK,
        python_callable=skip_promotion,
    )

    decide >> [promote, skip]
