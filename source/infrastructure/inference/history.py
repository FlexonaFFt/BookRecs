from __future__ import annotations

from typing import Any

from source.infrastructure.storage.postgres import PostgresClient


class UserHistoryProvider:
    def __init__(
        self,
        *,
        pg: PostgresClient | None,
        table_name: str = "user_item_interactions",
    ) -> None:
        self._pg = pg
        self._table_name = table_name
        self._bootstrap_done = False

    def get_seen_items(self, user_id: Any, limit: int = 500) -> set[Any]:
        if self._pg is None:
            return set()
        self._ensure_table()
        if limit <= 0:
            return set()
        query = (
            f"SELECT item_id FROM {self._table_name} "
            "WHERE user_id = %s "
            "ORDER BY interacted_at DESC "
            "LIMIT %s"
        )
        rows = self._pg.fetchall(query, (str(user_id), int(limit)))
        return {row["item_id"] for row in rows}

    def add_interaction(self, user_id: Any, item_id: Any, event_type: str = "implicit") -> None:
        if self._pg is None:
            return
        self._ensure_table()
        self._pg.execute(
            f"""
            INSERT INTO {self._table_name} (user_id, item_id, event_type)
            VALUES (%s, %s, %s)
            """,
            (str(user_id), str(item_id), str(event_type)),
        )

    def _ensure_table(self) -> None:
        if self._bootstrap_done:
            return
        self._pg.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                event_type TEXT NOT NULL DEFAULT 'implicit',
                interacted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        self._pg.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table_name}_user_time
            ON {self._table_name} (user_id, interacted_at DESC)
            """
        )
        self._bootstrap_done = True
