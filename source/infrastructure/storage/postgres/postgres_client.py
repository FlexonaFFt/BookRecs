from __future__ import annotations

from typing import Any

try:
    import psycopg
    from psycopg.rows import dict_row
except ModuleNotFoundError:
    psycopg = None
    dict_row = None
# Предоставляет низкоуровневые утилиты для выполнения запросов PostgreSQL.
class PostgresClient:
    """Small wrapper around psycopg for simple query operations."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def _ensure_driver(self) -> None:
        if psycopg is None:
            raise RuntimeError(
                "psycopg is required for PostgreSQL backend. Install dependency: pip install psycopg[binary]"
            )

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        self._ensure_driver()
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
            conn.commit()

    def fetchone(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        self._ensure_driver()
        with psycopg.connect(self._dsn, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                row = cur.fetchone()
        return row

    def fetchall(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        self._ensure_driver()
        with psycopg.connect(self._dsn, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        return list(rows)
