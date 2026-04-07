from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import psycopg


def _env_str(name: str, default: str) -> str:
    return str(os.getenv(name, default)).strip()


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


def _env_fraction(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    value = float(str(raw).strip())
    if not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0, 1], got {value}")
    return value


def _build_run_id(target_fraction: float) -> str:
    explicit = _env_str("BOOKRECS_ROLLBACK_RUN_ID", "")
    if explicit:
        return explicit
    pct = int(target_fraction * 100)
    return f"rollback_seed_{pct}pct"


def _load_train_rows(dataset_dir: str) -> int:
    train_path = Path(dataset_dir) / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Train parquet not found: {train_path}")
    train = pd.read_parquet(train_path, columns=["user_id"])
    return int(len(train))


def _count_rows(conn, event_type: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM user_item_interactions WHERE event_type = %s",
        (event_type,),
    ).fetchone()
    return int(row[0]) if row else 0


def _delete_rows_after_target(conn, *, event_type: str, target_count: int) -> int:
    result = conn.execute(
        """
        WITH doomed AS (
            SELECT id
            FROM user_item_interactions
            WHERE event_type = %s
            ORDER BY id ASC
            OFFSET %s
        )
        DELETE FROM user_item_interactions AS u
        USING doomed AS d
        WHERE u.id = d.id
        """,
        (event_type, target_count),
    )
    return int(result.rowcount or 0)


def _update_checkpoint(
    conn, *, run_id: str, interactions_count: int, reset_existing: bool
) -> None:
    if reset_existing:
        conn.execute("DELETE FROM training_checkpoints")

    conn.execute(
        """
        INSERT INTO training_checkpoints (run_id, interactions_count)
        VALUES (%s, %s)
        """,
        (run_id, interactions_count),
    )


def main() -> None:
    pg_dsn = _env_str("BOOKRECS_PG_DSN", "")
    if not pg_dsn:
        raise ValueError("BOOKRECS_PG_DSN is required")

    target_fraction = _env_fraction("BOOKRECS_ROLLBACK_TARGET_FRACTION", 0.25)
    dataset_dir = _env_str(
        "BOOKRECS_SEED_DATASET_DIR",
        "artifacts/tmp_preprocessed/goodreads_ya",
    )
    event_type = _env_str("BOOKRECS_ROLLBACK_EVENT_TYPE", "seed")
    dry_run = _env_bool("BOOKRECS_ROLLBACK_DRY_RUN", True)
    reset_checkpoints = _env_bool("BOOKRECS_ROLLBACK_RESET_CHECKPOINTS", True)
    run_id = _build_run_id(target_fraction)

    total_train_rows = _load_train_rows(dataset_dir)
    target_count = min(int(total_train_rows * target_fraction), total_train_rows)

    print(
        f"[rollback-seed] target_fraction={target_fraction:.1%}"
        f" target_count={target_count}"
        f" total_train_rows={total_train_rows}",
        flush=True,
    )
    print(
        f"[rollback-seed] event_type={event_type}"
        f" dry_run={dry_run}"
        f" reset_checkpoints={reset_checkpoints}",
        flush=True,
    )

    with psycopg.connect(pg_dsn) as conn:
        current_count = _count_rows(conn, event_type)
        print(f"[rollback-seed] current_count={current_count}", flush=True)

        if current_count <= target_count:
            print(
                "[rollback-seed] nothing to delete,"
                " current count is already below or equal to target",
                flush=True,
            )
            return

        to_delete = current_count - target_count
        if dry_run:
            print(
                f"[rollback-seed] dry-run: would delete {to_delete} rows"
                f" and write checkpoint run_id={run_id} count={target_count}",
                flush=True,
            )
            return

        deleted = _delete_rows_after_target(
            conn,
            event_type=event_type,
            target_count=target_count,
        )
        remaining = _count_rows(conn, event_type)
        _update_checkpoint(
            conn,
            run_id=run_id,
            interactions_count=remaining,
            reset_existing=reset_checkpoints,
        )
        conn.commit()

    print(
        f"[rollback-seed] done: deleted={deleted}"
        f" remaining={remaining}"
        f" checkpoint_run_id={run_id}",
        flush=True,
    )


if __name__ == "__main__":
    main()
