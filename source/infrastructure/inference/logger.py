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

    def log(self, payload: dict[str, Any]) -> None:
        if self._pg is None:
            return
        try:
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


def _validate_table_name(table_name: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name or ""):
        raise ValueError("Invalid PostgreSQL table name")
