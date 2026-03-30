from __future__ import annotations

import re
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
        _validate_table_name(table_name)
        self._table_name = table_name

    def get_seen_items(self, user_id: Any, limit: int = 500) -> set[Any]:
        if self._pg is None:
            return set()
        if limit <= 0:
            return set()
        query = (
            f"SELECT item_id FROM {self._table_name} "
            "WHERE user_id = %s "
            "ORDER BY interacted_at DESC "
            "LIMIT %s"
        )
        try:
            rows = self._pg.fetchall(query, (str(user_id), int(limit)))
            return {row["item_id"] for row in rows}
        except Exception:
            return set()

    def add_interaction(
        self, user_id: Any, item_id: Any, event_type: str = "implicit"
    ) -> None:
        if self._pg is None:
            return
        try:
            self._pg.execute(
                f"""
                INSERT INTO {self._table_name} (user_id, item_id, event_type)
                VALUES (%s, %s, %s)
                """,
                (str(user_id), str(item_id), str(event_type)),
            )
        except Exception:
            return


def _validate_table_name(table_name: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name or ""):
        raise ValueError("Invalid PostgreSQL table name")
