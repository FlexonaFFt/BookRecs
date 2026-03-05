from __future__ import annotations

from pathlib import Path

from source.infrastructure.storage.postgres import PostgresClient


# Применяет SQL-миграции в PostgreSQL.
def run_migration(pg_dsn: str, migration_path: str) -> None:
    path = Path(migration_path)
    if not path.exists():
        raise FileNotFoundError(f"Migration path not found: {path}")

    client = PostgresClient(pg_dsn)
    if path.is_file():
        _apply_sql(client=client, sql_path=path)
        return

    sql_files = sorted(path.glob("*.sql"))
    if not sql_files:
        raise ValueError(f"No .sql files found in migration dir: {path}")
    for sql_file in sql_files:
        _apply_sql(client=client, sql_path=sql_file)


def _apply_sql(client: PostgresClient, sql_path: Path) -> None:
    sql = sql_path.read_text(encoding="utf-8")
    statements = [stmt.strip() for stmt in sql.split(";") if stmt.strip()]
    for statement in statements:
        client.execute(statement)
