from __future__ import annotations

from pathlib import Path

from source.infrastructure.storage.postgres import PostgresClient


# Применяет SQL-миграции в PostgreSQL, пропуская уже применённые.
def run_migration(pg_dsn: str, migration_path: str) -> None:
    path = Path(migration_path)
    if not path.exists():
        raise FileNotFoundError(f"Migration path not found: {path}")

    client = PostgresClient(pg_dsn)
    _ensure_schema_migrations(client)

    if path.is_file():
        version = path.stem
        if not _is_applied(client, version):
            _apply_sql(client=client, sql_path=path)
            _record_applied(client, version)
            print(f"[migration] applied: {version}")
        else:
            print(f"[migration] skip (already applied): {version}")
        return

    sql_files = sorted(path.glob("*.sql"))
    if not sql_files:
        raise ValueError(f"No .sql files found in migration dir: {path}")
    for sql_file in sql_files:
        version = sql_file.stem
        if not _is_applied(client, version):
            _apply_sql(client=client, sql_path=sql_file)
            _record_applied(client, version)
            print(f"[migration] applied: {version}")
        else:
            print(f"[migration] skip (already applied): {version}")


def _ensure_schema_migrations(client: PostgresClient) -> None:
    client.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )


def _is_applied(client: PostgresClient, version: str) -> bool:
    row = client.fetchone(
        "SELECT version FROM schema_migrations WHERE version = %s",
        (version,),
    )
    return row is not None


def _record_applied(client: PostgresClient, version: str) -> None:
    client.execute(
        "INSERT INTO schema_migrations (version) VALUES (%s) ON CONFLICT DO NOTHING",
        (version,),
    )


def _apply_sql(client: PostgresClient, sql_path: Path) -> None:
    sql = sql_path.read_text(encoding="utf-8")
    statements = [stmt.strip() for stmt in sql.split(";") if stmt.strip()]
    for statement in statements:
        client.execute(statement)
