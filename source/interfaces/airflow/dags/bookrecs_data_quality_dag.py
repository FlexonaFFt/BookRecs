from __future__ import annotations

import json
import os
from datetime import datetime

import psycopg2 as psycopg
from airflow import DAG
from airflow.hooks.base import BaseHook
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from dag_common import (
    DEFAULT_ARGS,
    PG_CONN_ID,
    default_docker_args,
    docker_env,
    docker_secret_env,
)

_PROMOTE_TASK = "promote_model"
_SKIP_TASK = "skip_promotion"


def _pg_dsn() -> str:
    return BaseHook.get_connection(PG_CONN_ID).get_uri()


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
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT run_id, status, metrics_json, metadata_json
                FROM pipeline_runs
                WHERE pipeline_name = 'bookrecs'
                  AND status = 'SUCCESS'
                ORDER BY finished_at DESC
                LIMIT 3
                """
            )
            rows = cur.fetchall()

    if not rows:
        print("[data-quality] no successful runs found, skipping", flush=True)
        return _SKIP_TASK

    latest_run_id, _, metrics_json, _ = rows[0]
    context["ti"].xcom_push(key="run_id", value=latest_run_id)
    metrics = (
        json.loads(metrics_json)
        if isinstance(metrics_json, str)
        else (metrics_json or {})
    )

    ndcg = _get_float(metrics, "ndcg@10")
    recall = _get_float(metrics, "recall@10")

    print(
        f"[data-quality] latest run={latest_run_id}"
        f" ndcg@10={ndcg} recall@10={recall}",
        flush=True,
    )

    meets_threshold = (ndcg is None or ndcg >= min_ndcg) and (
        recall is None or recall >= min_recall
    )

    consecutive_skips = 0
    for _, _, _, m in rows[1:]:
        m_dict = json.loads(m) if isinstance(m, str) else (m or {})
        if not bool(m_dict.get("promoted", False)):
            consecutive_skips += 1

    force_promote = consecutive_skips >= force_after_skips

    if meets_threshold:
        print("[data-quality] metrics OK → promote", flush=True)
        context["ti"].xcom_push(key="promotion_reason", value="metrics_ok")
        return _PROMOTE_TASK

    if force_promote:
        print(
            f"[data-quality] skipped {consecutive_skips}x in a row → force promote",
            flush=True,
        )
        context["ti"].xcom_push(key="promotion_reason", value="force_after_skips")
        return _PROMOTE_TASK

    print(
        f"[data-quality] metrics below threshold"
        f" (ndcg={ndcg} < {min_ndcg} or recall={recall} < {min_recall}),"
        " skipping",
        flush=True,
    )
    context["ti"].xcom_push(key="promotion_reason", value="metrics_below_threshold")
    return _SKIP_TASK


def skip_promotion(**_) -> None:
    print("[data-quality] promotion skipped", flush=True)


def mark_promotion_result(*, promoted: bool, **context) -> None:
    dsn = _pg_dsn()
    run_id = context["ti"].xcom_pull(task_ids="decide_promotion", key="run_id")
    reason = context["ti"].xcom_pull(
        task_ids="decide_promotion", key="promotion_reason"
    )

    if not run_id:
        print("[data-quality] no run_id for metadata update, skipping", flush=True)
        return

    patch = {
        "promoted": bool(promoted),
        "promotion_reason": str(reason or ""),
        "promotion_checked_at": context.get("ts"),
    }

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE pipeline_runs
                SET metadata_json = COALESCE(metadata_json, '{}'::jsonb) || %s::jsonb,
                    updated_at = NOW()
                WHERE run_id = %s
                """,
                (json.dumps(patch), run_id),
            )
        conn.commit()

    print(
        f"[data-quality] metadata updated run_id={run_id} promoted={promoted}"
        f" reason={reason}",
        flush=True,
    )


def _promote_env() -> dict[str, str]:
    env = {**docker_env(), **docker_secret_env()}
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

    mark_promoted = PythonOperator(
        task_id="mark_promoted",
        python_callable=mark_promotion_result,
        op_kwargs={"promoted": True},
    )

    mark_skipped = PythonOperator(
        task_id="mark_skipped",
        python_callable=mark_promotion_result,
        op_kwargs={"promoted": False},
    )

    decide >> [promote, skip]
    promote >> mark_promoted
    skip >> mark_skipped
