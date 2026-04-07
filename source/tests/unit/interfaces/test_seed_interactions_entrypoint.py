from __future__ import annotations

from typing import Literal

import pandas as pd
import pytest

from source.interfaces import seed_interactions_entrypoint as mod


class _FakeResult:
    def __init__(self, row: tuple[int] | None = None) -> None:
        self._row = row

    def fetchone(self) -> tuple[int] | None:
        return self._row


class _FakeConn:
    def __init__(self, count: int) -> None:
        self.count = count

    def __enter__(self) -> _FakeConn:
        return self

    def __exit__(self, exc_type, exc, tb) -> Literal[False]:
        return False

    def execute(self, query: str, params=None):
        normalized = " ".join(query.split()).lower()
        if "select count(*) from user_item_interactions" in normalized:
            return _FakeResult((self.count,))
        raise AssertionError(f"Unexpected query: {query}")

    def commit(self) -> None:
        return None


def test_run_auto_rollback_restores_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOOKRECS_ROLLBACK_DRY_RUN", "true")

    captured: dict[str, str | None] = {}

    def _fake_rollback_main() -> None:
        captured["BOOKRECS_ROLLBACK_DRY_RUN"] = mod.os.getenv(
            "BOOKRECS_ROLLBACK_DRY_RUN"
        )
        captured["BOOKRECS_ROLLBACK_TARGET_FRACTION"] = mod.os.getenv(
            "BOOKRECS_ROLLBACK_TARGET_FRACTION"
        )
        captured["BOOKRECS_ROLLBACK_RESET_CHECKPOINTS"] = mod.os.getenv(
            "BOOKRECS_ROLLBACK_RESET_CHECKPOINTS"
        )

    monkeypatch.setattr(mod.rollback_seed_entrypoint, "main", _fake_rollback_main)

    mod._run_auto_rollback(
        pg_dsn="postgresql://demo",
        dataset_dir="/tmp/data",
        target_fraction=0.25,
    )

    assert captured["BOOKRECS_ROLLBACK_DRY_RUN"] == "false"
    assert captured["BOOKRECS_ROLLBACK_TARGET_FRACTION"] == "0.25"
    assert captured["BOOKRECS_ROLLBACK_RESET_CHECKPOINTS"] == "false"
    assert mod.os.getenv("BOOKRECS_ROLLBACK_DRY_RUN") == "true"


def test_main_triggers_auto_rollback_at_limit(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train.parquet").touch()

    monkeypatch.setenv("BOOKRECS_PG_DSN", "postgresql://demo")
    monkeypatch.setenv("BOOKRECS_SEED_DATASET_DIR", str(dataset_dir))
    monkeypatch.setenv("BOOKRECS_SEED_MAX_FRACTION", "0.75")
    monkeypatch.setenv("BOOKRECS_SEED_AUTO_ROLLBACK_ENABLED", "true")
    monkeypatch.setenv("BOOKRECS_SEED_AUTO_ROLLBACK_FRACTION", "0.25")
    monkeypatch.setattr(mod, "run_migration", lambda **_: None)
    monkeypatch.setattr(
        mod.pd,
        "read_parquet",
        lambda *_, **__: pd.DataFrame(
            {"user_id": [1, 2, 3, 4], "item_id": [10, 20, 30, 40]}
        ),
    )
    monkeypatch.setattr(mod.psycopg, "connect", lambda dsn: _FakeConn(3))

    called: dict[str, float] = {}

    def _fake_run_auto_rollback(
        *, pg_dsn: str, dataset_dir: str, target_fraction: float
    ) -> None:
        called["fraction"] = target_fraction

    monkeypatch.setattr(mod, "_run_auto_rollback", _fake_run_auto_rollback)

    mod.main()

    assert called["fraction"] == 0.25


def test_main_does_not_rollback_when_disabled(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train.parquet").touch()

    monkeypatch.setenv("BOOKRECS_PG_DSN", "postgresql://demo")
    monkeypatch.setenv("BOOKRECS_SEED_DATASET_DIR", str(dataset_dir))
    monkeypatch.setenv("BOOKRECS_SEED_MAX_FRACTION", "0.75")
    monkeypatch.setenv("BOOKRECS_SEED_AUTO_ROLLBACK_ENABLED", "false")
    monkeypatch.setattr(mod, "run_migration", lambda **_: None)
    monkeypatch.setattr(
        mod.pd,
        "read_parquet",
        lambda *_, **__: pd.DataFrame(
            {"user_id": [1, 2, 3, 4], "item_id": [10, 20, 30, 40]}
        ),
    )
    monkeypatch.setattr(mod.psycopg, "connect", lambda dsn: _FakeConn(3))

    called = {"rollback": 0}

    def _fake_run_auto_rollback(**kwargs) -> None:
        called["rollback"] += 1

    monkeypatch.setattr(mod, "_run_auto_rollback", _fake_run_auto_rollback)

    mod.main()

    assert called["rollback"] == 0
