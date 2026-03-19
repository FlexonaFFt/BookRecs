from __future__ import annotations

import json

from source.application.ports import RunLogPort
from source.domain.entities import PipelineRun
from source.infrastructure.storage.postgres.postgres_client import PostgresClient


# Сохраняет события запуска пайплайна в PostgreSQL.
class PostgresRunLog(RunLogPort):

    def __init__(self, pg: PostgresClient) -> None:
        self._pg = pg

    def start(self, run: PipelineRun) -> None:
        self._pg.execute(
            """
            INSERT INTO pipeline_runs (
                run_id, pipeline_name, status, started_at, finished_at,
                message, metrics_json, metadata_json, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, NOW())
            ON CONFLICT (run_id) DO UPDATE SET
                pipeline_name = EXCLUDED.pipeline_name,
                status = EXCLUDED.status,
                started_at = EXCLUDED.started_at,
                finished_at = EXCLUDED.finished_at,
                message = EXCLUDED.message,
                metrics_json = EXCLUDED.metrics_json,
                metadata_json = EXCLUDED.metadata_json,
                updated_at = NOW()
            """,
            (
                run.run_id,
                run.pipeline_name,
                run.status.value,
                run.started_at,
                run.finished_at,
                run.message,
                json.dumps(run.metrics, ensure_ascii=False),
                json.dumps(run.metadata, ensure_ascii=False),
            ),
        )

    def finish(self, run: PipelineRun) -> None:
        self._pg.execute(
            """
            UPDATE pipeline_runs
            SET status = %s,
                finished_at = %s,
                message = %s,
                metrics_json = %s::jsonb,
                metadata_json = %s::jsonb,
                updated_at = NOW()
            WHERE run_id = %s
            """,
            (
                run.status.value,
                run.finished_at,
                run.message,
                json.dumps(run.metrics, ensure_ascii=False),
                json.dumps(run.metadata, ensure_ascii=False),
                run.run_id,
            ),
        )
