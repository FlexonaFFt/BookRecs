from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import psycopg

from source.interfaces.migration_runner import run_migration


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name, default)
    return str(raw).strip()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    value = float(str(raw).strip())
    if not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0, 1], got {value}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be boolean-like, got: {raw}")


def main() -> None:
    pg_dsn = _env_str("BOOKRECS_PG_DSN", "")
    if not pg_dsn:
        raise ValueError("BOOKRECS_PG_DSN is required")

    dataset_dir = Path(
        _env_str(
            "BOOKRECS_SEED_DATASET_DIR",
            "artifacts/tmp_preprocessed/goodreads_ya",
        )
    )
    batch_fraction = _env_float("BOOKRECS_SEED_BATCH_FRACTION", 0.15)
    max_fraction = _env_float("BOOKRECS_SEED_MAX_FRACTION", 0.75)
    chunk_size = 500
    migration_path = _env_str(
        "BOOKRECS_PG_MIGRATION_PATH",
        "source/infrastructure/storage/postgres/migrations",
    )

    if _env_bool("BOOKRECS_RUN_MIGRATE", True):
        run_migration(pg_dsn=pg_dsn, migration_path=migration_path)

    train_path = dataset_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Train parquet not found: {train_path}")

    print(f"[seed] loading {train_path}", flush=True)
    train = pd.read_parquet(train_path, columns=["user_id", "item_id"])
    train = train.sort_values(["user_id", "item_id"]).reset_index(drop=True)
    total = len(train)
    max_rows = int(total * max_fraction)
    batch_rows = int(total * batch_fraction)

    print(
        f"[seed] total={total} max_rows={max_rows} batch_rows={batch_rows}",
        flush=True,
    )

    with psycopg.connect(pg_dsn) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM user_item_interactions WHERE event_type = 'seed'"
        ).fetchone()
        current_count = int(row[0]) if row else 0

    print(f"[seed] already_seeded={current_count}", flush=True)

    if current_count >= max_rows:
        print(
            f"[seed] max fraction {max_fraction:.0%} reached ({current_count} rows),"
            " skipping",
            flush=True,
        )
        return

    offset = current_count
    end = min(offset + batch_rows, max_rows)
    batch = train.iloc[offset:end]

    if batch.empty:
        print("[seed] nothing to insert", flush=True)
        return

    rows = [
        (str(r.user_id), str(r.item_id), "seed") for r in batch.itertuples(index=False)
    ]

    print(
        f"[seed] inserting {len(rows)} interactions"
        f" (slice {offset}→{end}, {end / total:.1%} of dataset)",
        flush=True,
    )

    insert_prefix = (
        "INSERT INTO user_item_interactions (user_id, item_id, event_type) VALUES"
    )
    placeholder = "(%s, %s, %s)"

    with psycopg.connect(pg_dsn) as conn:
        for i in range(0, len(rows), chunk_size):
            part = rows[i : i + chunk_size]
            values = ", ".join([placeholder] * len(part))
            flat = [v for r in part for v in r]
            conn.execute(f"{insert_prefix} {values}", flat)
            conn.commit()
            done = min(i + chunk_size, len(rows))
            print(f"[seed] progress {done}/{len(rows)}", flush=True)

    print(
        f"[seed] done: added {len(rows)} interactions,"
        f" total seeded={end} ({end / total:.1%} of dataset)",
        flush=True,
    )


if __name__ == "__main__":
    main()
