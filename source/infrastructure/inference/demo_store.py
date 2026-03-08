from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from source.infrastructure.storage.postgres import PostgresClient


@dataclass(frozen=True)
class DemoUser:
    user_id: str
    history_len: int


@dataclass(frozen=True)
class DemoBook:
    item_id: int
    title: str
    description: str
    url: str
    image_url: str
    authors: list[str]
    tags: list[str]
    series: list[str]


class DemoStore:
    def __init__(self, *, pg: PostgresClient | None) -> None:
        self._pg = pg

    def list_users(self, limit: int = 100) -> list[DemoUser]:
        if self._pg is None:
            return []
        safe_limit = min(max(1, int(limit)), 5000)
        rows = self._pg.fetchall(
            """
            SELECT user_id, history_len
            FROM demo_users
            ORDER BY history_len DESC, user_id ASC
            LIMIT %s
            """,
            (safe_limit,),
        )
        return [
            DemoUser(user_id=str(row["user_id"]), history_len=int(row.get("history_len", 0) or 0))
            for row in rows
        ]

    def list_books(
        self,
        *,
        limit: int = 40,
        offset: int = 0,
        q: str = "",
        genre: str = "",
    ) -> tuple[list[DemoBook], int]:
        if self._pg is None:
            return [], 0

        safe_limit = min(max(1, int(limit)), 100)
        safe_offset = max(0, int(offset))
        safe_q = (q or "").strip().lower()
        safe_genre = (genre or "").strip().lower()

        where_parts: list[str] = []
        params: list[Any] = []

        if safe_q:
            where_parts.append("LOWER(title) LIKE %s")
            params.append(f"%{safe_q}%")

        if safe_genre and safe_genre != "all":
            where_parts.append("EXISTS (SELECT 1 FROM jsonb_array_elements_text(tags_json) AS t(tag) WHERE LOWER(t.tag) = %s)")
            params.append(safe_genre)

        where_sql = ""
        if where_parts:
            where_sql = "WHERE " + " AND ".join(where_parts)

        count_row = self._pg.fetchone(
            f"SELECT COUNT(*) AS total FROM demo_books {where_sql}",
            tuple(params),
        )
        total = int((count_row or {}).get("total", 0) or 0)

        query = f"""
            SELECT item_id, title, description, url, image_url, authors_json, tags_json, series_json
            FROM demo_books
            {where_sql}
            ORDER BY item_id ASC
            LIMIT %s OFFSET %s
        """
        rows = self._pg.fetchall(query, tuple([*params, safe_limit, safe_offset]))
        return [self._map_book(row) for row in rows], total

    def get_book(self, item_id: int) -> DemoBook | None:
        if self._pg is None:
            return None
        row = self._pg.fetchone(
            """
            SELECT item_id, title, description, url, image_url, authors_json, tags_json, series_json
            FROM demo_books
            WHERE item_id = %s
            """,
            (int(item_id),),
        )
        if not row:
            return None
        return self._map_book(row)

    @staticmethod
    def _map_book(row: dict[str, Any]) -> DemoBook:
        def _as_list(value: Any) -> list[str]:
            if isinstance(value, list):
                return [str(x) for x in value]
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return []
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return [str(x) for x in parsed]
                except Exception:
                    return []
            return []

        return DemoBook(
            item_id=int(row["item_id"]),
            title=str(row.get("title", "") or ""),
            description=str(row.get("description", "") or ""),
            url=str(row.get("url", "") or ""),
            image_url=str(row.get("image_url", "") or ""),
            authors=_as_list(row.get("authors_json")),
            tags=_as_list(row.get("tags_json")),
            series=_as_list(row.get("series_json")),
        )
