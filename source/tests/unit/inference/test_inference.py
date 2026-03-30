from __future__ import annotations

from typing import Any

import pytest

from source.infrastructure.inference.history import UserHistoryProvider
from source.infrastructure.inference.logger import InferenceRequestLogger


class FakePostgresClient:
    def __init__(self) -> None:
        self.executions: list[tuple[str, tuple]] = []
        self._fetchall_result: list[dict[str, Any]] = []

    def execute(self, query: str, params: tuple = ()) -> None:
        self.executions.append((query.strip(), params))

    def fetchone(self, query: str, params: tuple = ()) -> dict[str, Any] | None:
        return None

    def fetchall(self, query: str, params: tuple = ()) -> list[dict[str, Any]]:
        self.executions.append((query.strip(), params))
        return self._fetchall_result


# --- table name validation ---


@pytest.mark.parametrize(
    "table_name",
    ["valid_name", "valid_name_2", "A_table"],
)
def test_table_name_validation_accepts_safe_identifiers(table_name: str) -> None:
    _ = UserHistoryProvider(pg=None, table_name=table_name)
    _ = InferenceRequestLogger(pg=None, table_name=table_name)


@pytest.mark.parametrize(
    "table_name",
    ["bad-name", "bad name", "123table", "a;drop table x", ""],
)
def test_table_name_validation_rejects_unsafe_identifiers(table_name: str) -> None:
    with pytest.raises(ValueError):
        _ = UserHistoryProvider(pg=None, table_name=table_name)
    with pytest.raises(ValueError):
        _ = InferenceRequestLogger(pg=None, table_name=table_name)


# --- UserHistoryProvider: pg=None ---


def test_get_seen_items_returns_empty_when_pg_is_none() -> None:
    provider = UserHistoryProvider(pg=None)
    assert provider.get_seen_items("user1") == set()


def test_get_seen_items_returns_empty_when_limit_is_zero() -> None:
    fake = FakePostgresClient()
    provider = UserHistoryProvider(pg=fake)  # type: ignore[arg-type]
    assert provider.get_seen_items("user1", limit=0) == set()
    assert not fake.executions


def test_add_interaction_does_nothing_when_pg_is_none() -> None:
    provider = UserHistoryProvider(pg=None)
    provider.add_interaction("user1", "item1")  # should not raise


def test_get_seen_items_returns_item_ids_from_db() -> None:
    fake = FakePostgresClient()
    fake._fetchall_result = [{"item_id": "book1"}, {"item_id": "book2"}]
    provider = UserHistoryProvider(pg=fake)  # type: ignore[arg-type]
    result = provider.get_seen_items("user1")
    assert result == {"book1", "book2"}


def test_add_interaction_executes_insert() -> None:
    fake = FakePostgresClient()
    provider = UserHistoryProvider(pg=fake)  # type: ignore[arg-type]
    provider.add_interaction("user1", "book1", "click")
    assert len(fake.executions) == 1
    _, params = fake.executions[0]
    assert params == ("user1", "book1", "click")


# --- InferenceRequestLogger: pg=None ---


def test_log_does_nothing_when_pg_is_none() -> None:
    logger = InferenceRequestLogger(pg=None)
    logger.log({"user_id": "u1", "endpoint": "/recommend"})  # should not raise


def test_log_executes_insert_with_correct_fields() -> None:
    fake = FakePostgresClient()
    logger = InferenceRequestLogger(pg=fake)  # type: ignore[arg-type]
    logger.log(
        {
            "user_id": "u1",
            "endpoint": "/recommend",
            "request": {"k": 10},
            "response": {"items": []},
            "model_dir": "artifacts/runs/run1",
            "latency_ms": 42,
        }
    )
    assert len(fake.executions) == 1
    _, params = fake.executions[0]
    assert params[0] == "u1"
    assert params[1] == "/recommend"
    assert params[4] == "artifacts/runs/run1"
    assert params[5] == 42


# --- no _ensure_table calls ---


def test_history_provider_has_no_ensure_table_method() -> None:
    provider = UserHistoryProvider(pg=None)
    assert not hasattr(provider, "_ensure_table")
    assert not hasattr(provider, "_bootstrap_done")


def test_inference_logger_has_no_ensure_table_method() -> None:
    logger = InferenceRequestLogger(pg=None)
    assert not hasattr(logger, "_ensure_table")
    assert not hasattr(logger, "_bootstrap_done")
