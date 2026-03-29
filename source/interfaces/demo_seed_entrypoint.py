from __future__ import annotations

import json
import os
from datetime import timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import psycopg

from source.interfaces.migration_runner import run_migration


def env_str(name: str, default: str) -> str:
    raw = os.getenv(name, default)
    return str(raw).strip()


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    value = int(str(raw).strip())
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be boolean-like, got: {raw}")


def chunked(rows: Sequence[tuple], chunk_size: int) -> Iterable[Sequence[tuple]]:
    total = len(rows)
    for i in range(0, total, chunk_size):
        yield rows[i : i + chunk_size]


def _insert_chunk(
    conn,
    insert_prefix: str,
    row_placeholder: str,
    rows: Sequence[tuple],
    conflict_suffix: str = "",
) -> None:
    values = ", ".join([row_placeholder] * len(rows))
    flat = [v for row in rows for v in row]
    conn.execute(f"{insert_prefix} {values} {conflict_suffix}", flat)


def to_json_list(value: object) -> str:
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return json.dumps([], ensure_ascii=False)


def to_event_ts(value: object):
    ts = pd.to_datetime(value, errors="coerce", utc=False)
    if pd.isna(ts):
        return None
    py_dt = ts.to_pydatetime()
    if py_dt.tzinfo is None:
        return py_dt.replace(tzinfo=timezone.utc)
    return py_dt.astimezone(timezone.utc)


def main() -> None:
    pg_dsn = env_str("BOOKRECS_PG_DSN", "")
    if not pg_dsn:
        raise ValueError("BOOKRECS_PG_DSN is required")

    dataset_dir = Path(
        env_str("BOOKRECS_DEMO_DATASET_DIR", "artifacts/tmp_preprocessed/goodreads_ya")
    )
    users_limit = env_int("BOOKRECS_DEMO_USERS_LIMIT", 2000)
    max_history_per_user = env_int("BOOKRECS_DEMO_MAX_HISTORY_PER_USER", 100)
    reset = env_bool("BOOKRECS_DEMO_RESET", True)
    migration_path = env_str(
        "BOOKRECS_PG_MIGRATION_PATH",
        "source/infrastructure/storage/postgres/migrations",
    )

    books_path = dataset_dir / "books.parquet"
    train_path = dataset_dir / "train.parquet"

    if not books_path.exists():
        raise FileNotFoundError(f"Books parquet not found: {books_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"Train parquet not found: {train_path}")

    run_migrate = env_bool("BOOKRECS_RUN_MIGRATE", True)
    if run_migrate:
        print(f"[demo-seed] run migrations from {migration_path}")
        run_migration(pg_dsn=pg_dsn, migration_path=migration_path)
    else:
        print("[demo-seed] skip migrations (BOOKRECS_RUN_MIGRATE=false)")

    print(f"[demo-seed] read books from {books_path}")
    books = pd.read_parquet(books_path)
    required_books = {"item_id", "title"}
    missing_books = required_books - set(books.columns)
    if missing_books:
        raise ValueError(f"books.parquet missing columns: {sorted(missing_books)}")

    print(f"[demo-seed] read train interactions from {train_path}")
    train = pd.read_parquet(train_path)
    required_train = {"user_id", "item_id", "date_added"}
    missing_train = required_train - set(train.columns)
    if missing_train:
        raise ValueError(f"train.parquet missing columns: {sorted(missing_train)}")

    user_counts = (
        train.groupby("user_id")["item_id"].size().sort_values(ascending=False)
    )
    selected_users = user_counts.head(users_limit).index
    filtered = train[train["user_id"].isin(selected_users)].copy()

    filtered["date_added"] = pd.to_datetime(filtered["date_added"], errors="coerce")
    filtered = filtered.dropna(subset=["date_added"])
    filtered = filtered.sort_values(["user_id", "date_added"], ascending=[True, False])
    filtered = (
        filtered.groupby("user_id", as_index=False).head(max_history_per_user).copy()
    )

    history_len = (
        filtered.groupby("user_id")["item_id"]
        .size()
        .rename("history_len")
        .reset_index()
    )

    book_rows = []
    for row in books.itertuples(index=False):
        item_id = int(getattr(row, "item_id"))
        book_rows.append(
            (
                item_id,
                str(getattr(row, "title", "") or ""),
                str(getattr(row, "description", "") or ""),
                str(getattr(row, "url", "") or ""),
                str(getattr(row, "image_url", "") or ""),
                to_json_list(getattr(row, "authors", [])),
                to_json_list(getattr(row, "tags", [])),
                to_json_list(getattr(row, "series", [])),
            )
        )

    user_rows = [
        (str(row.user_id), int(row.history_len))
        for row in history_len.itertuples(index=False)
    ]

    seen_rows = []
    for row in filtered.itertuples(index=False):
        event_ts = to_event_ts(getattr(row, "date_added"))
        if event_ts is None:
            continue
        seen_rows.append(
            (
                str(getattr(row, "user_id")),
                int(getattr(row, "item_id")),
                "seed",
                event_ts,
            )
        )

    print(
        "[demo-seed] prepared rows: "
        f"books={len(book_rows)} users={len(user_rows)} seen={len(seen_rows)}"
    )

    book_insert = (
        "INSERT INTO demo_books"
        " (item_id, title, description, url, image_url,"
        "  authors_json, tags_json, series_json) VALUES"
    )
    book_row = "(%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)"
    book_conflict = (
        "ON CONFLICT (item_id) DO UPDATE SET"
        " title=EXCLUDED.title, description=EXCLUDED.description,"
        " url=EXCLUDED.url, image_url=EXCLUDED.image_url,"
        " authors_json=EXCLUDED.authors_json,"
        " tags_json=EXCLUDED.tags_json, series_json=EXCLUDED.series_json"
    )

    user_insert = "INSERT INTO demo_users (user_id, history_len) VALUES"
    user_row = "(%s, %s)"
    user_conflict = (
        "ON CONFLICT (user_id) DO UPDATE SET"
        " history_len=EXCLUDED.history_len, updated_at=NOW()"
    )

    seen_insert = (
        "INSERT INTO demo_user_seen (user_id, item_id, event_type, event_ts) VALUES"
    )
    seen_row = "(%s, %s, %s, %s)"
    seen_conflict = (
        "ON CONFLICT (user_id, item_id) DO UPDATE SET"
        " event_type=EXCLUDED.event_type, event_ts=EXCLUDED.event_ts"
    )

    chunk_size = 500

    with psycopg.connect(pg_dsn) as conn:
        if reset:
            print("[demo-seed] truncate demo tables")
            conn.execute("TRUNCATE TABLE demo_user_seen, demo_users, demo_books")
            conn.commit()

        chunks = list(chunked(book_rows, chunk_size))
        for i, part in enumerate(chunks, 1):
            _insert_chunk(conn, book_insert, book_row, part, book_conflict)
            conn.commit()
            done = min(i * chunk_size, len(book_rows))
            print(f"[demo-seed] books {i}/{len(chunks)} ({done}/{len(book_rows)})")

        chunks = list(chunked(user_rows, chunk_size))
        for i, part in enumerate(chunks, 1):
            _insert_chunk(conn, user_insert, user_row, part, user_conflict)
            conn.commit()
            done = min(i * chunk_size, len(user_rows))
            print(f"[demo-seed] users {i}/{len(chunks)} ({done}/{len(user_rows)})")

        chunks = list(chunked(seen_rows, chunk_size))
        for i, part in enumerate(chunks, 1):
            _insert_chunk(conn, seen_insert, seen_row, part, seen_conflict)
            conn.commit()
            done = min(i * chunk_size, len(seen_rows))
            print(f"[demo-seed] seen {i}/{len(chunks)} ({done}/{len(seen_rows)})")

    print("[demo-seed] done")


if __name__ == "__main__":
    main()
