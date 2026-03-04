from __future__ import annotations

import os
from pathlib import Path

from source.infrastructure.storage.postgres import PostgresClient


def _apply_sql(client: PostgresClient, sql_path: Path) -> None:
    sql = sql_path.read_text(encoding="utf-8")
    statements = [stmt.strip() for stmt in sql.split(";") if stmt.strip()]
    for statement in statements:
        client.execute(statement)


def run_migration(pg_dsn: str, migration_path: str) -> None:
    path = Path(migration_path)
    if not path.exists():
        raise FileNotFoundError(f"Migration path not found: {path}")

    client = PostgresClient(pg_dsn)
    if path.is_file():
        _apply_sql(client, path)
        return

    sql_files = sorted(path.glob("*.sql"))
    if not sql_files:
        raise ValueError(f"No .sql files found in migration dir: {path}")
    for sql_file in sql_files:
        _apply_sql(client, sql_file)


def main() -> None:
    dsn = os.getenv("BOOKRECS_PG_DSN", "").strip()
    migration_file = os.getenv(
        "BOOKRECS_PG_MIGRATION_PATH",
        "source/infrastructure/storage/postgres/migrations",
    ).strip()
    if not dsn:
        raise ValueError("BOOKRECS_PG_DSN is required for migration")
    run_migration(dsn, migration_file)
    print(f"Migration(s) applied from: {migration_file}")


if __name__ == "__main__":
    main()
