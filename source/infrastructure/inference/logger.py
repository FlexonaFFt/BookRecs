from __future__ import annotations

import json
import re
from typing import Any

from source.infrastructure.storage.postgres import PostgresClient


class InferenceRequestLogger:
    def __init__(
        self,
        *,
        pg: PostgresClient | None,
        table_name: str = "inference_requests",
    ) -> None:
        self._pg = pg
        _validate_table_name(table_name)
        self._table_name = table_name
        self._bootstrap_done = False

    def log(self, payload: dict[str, Any]) -> None:
        if self._pg is None:
            return
        try:
            self._ensure_table()
            self._pg.execute(
                f"""
                INSERT INTO {self._table_name} (
                    user_id, endpoint,
                    request_json, response_json,
                    model_dir, latency_ms
                )
                VALUES (%s, %s, %s::jsonb, %s::jsonb, %s, %s)
                """,
                (
                    str(payload.get("user_id", "")),
                    str(payload.get("endpoint", "")),
                    json.dumps(payload.get("request", {}), ensure_ascii=False),
                    json.dumps(payload.get("response", {}), ensure_ascii=False),
                    str(payload.get("model_dir", "")),
                    int(payload.get("latency_ms", 0)),
                ),
            )
        except Exception:
            return

    def _ensure_table(self) -> None:
        if self._bootstrap_done:
            return
        if self._pg is None:
            return
        self._pg.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                request_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                response_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                model_dir TEXT NOT NULL DEFAULT '',
                latency_ms INT NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        self._pg.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table_name}_endpoint_time
            ON {self._table_name} (endpoint, created_at DESC)
            """
        )
        self._bootstrap_done = True


def _validate_table_name(table_name: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name or ""):
        raise ValueError("Invalid PostgreSQL table name")
