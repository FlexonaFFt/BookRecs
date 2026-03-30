from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from source.interfaces import migration_runner as mod


class FakePostgresClient:
    def __init__(self) -> None:
        self.executions: list[tuple[str, tuple]] = []
        self._fetchone_result: dict[str, Any] | None = None

    def execute(self, query: str, params: tuple = ()) -> None:
        self.executions.append((query.strip(), params))

    def fetchone(self, query: str, params: tuple = ()) -> dict[str, Any] | None:
        self.executions.append((query.strip(), params))
        return self._fetchone_result

    def fetchall(self, query: str, params: tuple = ()) -> list[dict[str, Any]]:
        self.executions.append((query.strip(), params))
        return []

    def executed_queries(self) -> list[str]:
        return [q for q, _ in self.executions]


def _make_fake_client(monkeypatch: pytest.MonkeyPatch) -> FakePostgresClient:
    fake = FakePostgresClient()
    monkeypatch.setattr(mod, "PostgresClient", lambda dsn: fake)
    return fake


# --- _ensure_schema_migrations ---


def test_ensure_schema_migrations_creates_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _make_fake_client(monkeypatch)
    mod._ensure_schema_migrations(fake)
    assert any("schema_migrations" in q for q in fake.executed_queries())


# --- _is_applied ---


def test_is_applied_returns_true_when_row_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _make_fake_client(monkeypatch)
    fake._fetchone_result = {"version": "0001_init"}
    assert mod._is_applied(fake, "0001_init") is True


def test_is_applied_returns_false_when_no_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _make_fake_client(monkeypatch)
    fake._fetchone_result = None
    assert mod._is_applied(fake, "0001_init") is False


# --- _record_applied ---


def test_record_applied_inserts_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _make_fake_client(monkeypatch)
    mod._record_applied(fake, "0001_init")
    params_list = [p for _, p in fake.executions]
    assert ("0001_init",) in params_list


# --- run_migration: single file ---


def test_run_migration_applies_new_single_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sql_file = tmp_path / "0001_init.sql"
    sql_file.write_text("CREATE TABLE foo (id INT)", encoding="utf-8")

    fake = _make_fake_client(monkeypatch)
    fake._fetchone_result = None  # not yet applied

    mod.run_migration("dummy_dsn", str(sql_file))

    queries = fake.executed_queries()
    assert any("CREATE TABLE foo" in q for q in queries)
    assert any("INSERT INTO schema_migrations" in q for q in queries)


def test_run_migration_skips_already_applied_single_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sql_file = tmp_path / "0001_init.sql"
    sql_file.write_text("CREATE TABLE foo (id INT)", encoding="utf-8")

    fake = _make_fake_client(monkeypatch)
    fake._fetchone_result = {"version": "0001_init"}  # already applied

    mod.run_migration("dummy_dsn", str(sql_file))

    queries = fake.executed_queries()
    assert not any("CREATE TABLE foo" in q for q in queries)
    assert not any("INSERT INTO schema_migrations" in q for q in queries)


# --- run_migration: directory ---


def test_run_migration_applies_only_new_files_in_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "0001_init.sql").write_text(
        "CREATE TABLE a (id INT)", encoding="utf-8"
    )
    (tmp_path / "0002_more.sql").write_text(
        "CREATE TABLE b (id INT)", encoding="utf-8"
    )

    fake = _make_fake_client(monkeypatch)

    call_count = 0

    def _fetchone_side_effect(query: str, params: tuple = ()) -> dict | None:
        nonlocal call_count
        fake.executions.append((query.strip(), params))
        # First migration already applied, second is new
        if params and params[0] == "0001_init":
            return {"version": "0001_init"}
        call_count += 1
        return None

    fake.fetchone = _fetchone_side_effect  # type: ignore[method-assign]

    mod.run_migration("dummy_dsn", str(tmp_path))

    queries = fake.executed_queries()
    assert not any("CREATE TABLE a" in q for q in queries)
    assert any("CREATE TABLE b" in q for q in queries)


def test_run_migration_raises_if_path_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        mod.run_migration("dummy_dsn", "/nonexistent/path")


def test_run_migration_raises_if_dir_has_no_sql(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_fake_client(monkeypatch)
    with pytest.raises(ValueError, match="No .sql files"):
        mod.run_migration("dummy_dsn", str(tmp_path))


# --- _apply_sql splits statements correctly ---


def test_apply_sql_splits_multiple_statements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sql_file = tmp_path / "multi.sql"
    sql_file.write_text(
        "CREATE TABLE x (id INT);\nCREATE TABLE y (id INT);",
        encoding="utf-8",
    )
    fake = _make_fake_client(monkeypatch)
    mod._apply_sql(fake, sql_file)

    queries = fake.executed_queries()
    assert any("CREATE TABLE x" in q for q in queries)
    assert any("CREATE TABLE y" in q for q in queries)
