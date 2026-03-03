from __future__ import annotations

from pathlib import Path

from source.infrastructure.storage import ClientPg


def run_migration(pg_dsn: str, migration_file: str) -> None:
    path = Path(migration_file)
    if not path.exists():
        raise FileNotFoundError(f"Migration file not found: {path}")

    sql = path.read_text(encoding="utf-8")
    statements = [stmt.strip() for stmt in sql.split(";") if stmt.strip()]
    client = ClientPg(pg_dsn)
    for statement in statements:
        client.execute(statement)


if __name__ == "__main__":
    import os

    dsn = os.getenv("BOOKRECS_PG_DSN", "").strip()
    migration_file = os.getenv("BOOKRECS_PG_MIGRATION_FILE", "configs/sql/001_init.sql").strip()
    if not dsn:
        raise ValueError("BOOKRECS_PG_DSN is required for migration")
    run_migration(dsn, migration_file)
    print(f"Migration applied: {migration_file}")
