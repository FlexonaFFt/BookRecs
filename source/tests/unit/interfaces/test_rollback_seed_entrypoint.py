from __future__ import annotations

from typing import Literal

import pytest

from source.interfaces import rollback_seed_entrypoint as mod


class _FakeResult:
    def __init__(self, row: tuple[int] | None = None, rowcount: int = 0) -> None:
        self._row = row
        self.rowcount = rowcount

    def fetchone(self) -> tuple[int] | None:
        return self._row


class _FakeConn:
    def __init__(self, start_count: int) -> None:
        self.current_count = start_count
        self.deleted = 0
        self.checkpoints: list[tuple[str, int]] = []
        self.reset_called = False
        self.commit_called = False

    def __enter__(self) -> _FakeConn:
        return self

    def __exit__(self, exc_type, exc, tb) -> Literal[False]:
        return False

    def execute(self, query: str, params: tuple = ()) -> _FakeResult:
        normalized = " ".join(query.split()).lower()
        if "select count(*) from user_item_interactions" in normalized:
            return _FakeResult((self.current_count,))
        if normalized.startswith("with doomed as ("):
            target_count = int(params[1])
            self.deleted = max(0, self.current_count - target_count)
            self.current_count = target_count
            return _FakeResult(rowcount=self.deleted)
        if normalized.startswith("delete from training_checkpoints"):
            self.reset_called = True
            self.checkpoints.clear()
            return _FakeResult()
        if normalized.startswith("insert into training_checkpoints"):
            run_id = str(params[0])
            count = int(params[1])
            self.checkpoints.append((run_id, count))
            return _FakeResult()
        raise AssertionError(f"Unexpected query: {query}")

    def commit(self) -> None:
        self.commit_called = True


def test_env_fraction_validates_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOOKRECS_ROLLBACK_TARGET_FRACTION", "1.5")
    with pytest.raises(ValueError):
        mod._env_fraction("BOOKRECS_ROLLBACK_TARGET_FRACTION", 0.25)


def test_build_run_id_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOOKRECS_ROLLBACK_RUN_ID", raising=False)
    assert mod._build_run_id(0.25) == "rollback_seed_25pct"


def test_main_dry_run_does_not_mutate(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeConn(start_count=100)

    monkeypatch.setenv("BOOKRECS_PG_DSN", "postgresql://demo")
    monkeypatch.setenv("BOOKRECS_ROLLBACK_TARGET_FRACTION", "0.25")
    monkeypatch.setenv("BOOKRECS_ROLLBACK_DRY_RUN", "true")
    monkeypatch.setenv("BOOKRECS_SEED_DATASET_DIR", "/tmp/unused")
    monkeypatch.setattr(mod, "_load_train_rows", lambda _: 200)
    monkeypatch.setattr(mod.psycopg, "connect", lambda dsn: fake)

    mod.main()

    assert fake.current_count == 100
    assert fake.deleted == 0
    assert fake.checkpoints == []
    assert fake.commit_called is False


def test_main_rolls_back_and_writes_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeConn(start_count=180)

    monkeypatch.setenv("BOOKRECS_PG_DSN", "postgresql://demo")
    monkeypatch.setenv("BOOKRECS_ROLLBACK_TARGET_FRACTION", "0.25")
    monkeypatch.setenv("BOOKRECS_ROLLBACK_DRY_RUN", "false")
    monkeypatch.setenv("BOOKRECS_ROLLBACK_RESET_CHECKPOINTS", "true")
    monkeypatch.setenv("BOOKRECS_SEED_DATASET_DIR", "/tmp/unused")
    monkeypatch.setattr(mod, "_load_train_rows", lambda _: 400)
    monkeypatch.setattr(mod.psycopg, "connect", lambda dsn: fake)

    mod.main()

    assert fake.deleted == 80
    assert fake.current_count == 100
    assert fake.reset_called is True
    assert fake.checkpoints == [("rollback_seed_25pct", 100)]
    assert fake.commit_called is True
