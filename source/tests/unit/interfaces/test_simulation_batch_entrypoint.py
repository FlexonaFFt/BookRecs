from __future__ import annotations

import os

import pytest

from source.interfaces import simulation_batch_entrypoint as mod


def test_main_sets_run_name_from_execution_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BOOKRECS_BATCH_EXECUTION_DATE", "2026-03-16")
    monkeypatch.delenv("BOOKRECS_BATCH_RUN_NAME", raising=False)

    called: dict[str, int] = {"batch": 0}

    def _fake_batch() -> None:
        called["batch"] += 1

    monkeypatch.setattr(mod, "_run_batch", _fake_batch)
    mod.main()

    assert called["batch"] == 1
    assert os.environ.get("BOOKRECS_BATCH_RUN_NAME") == "simulation_20260316"


def test_main_uses_manual_when_no_execution_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BOOKRECS_BATCH_EXECUTION_DATE", raising=False)
    monkeypatch.delenv("BOOKRECS_BATCH_RUN_NAME", raising=False)

    called: dict[str, int] = {"batch": 0}

    def _fake_batch() -> None:
        called["batch"] += 1

    monkeypatch.setattr(mod, "_run_batch", _fake_batch)
    mod.main()

    assert called["batch"] == 1
    assert os.environ.get("BOOKRECS_BATCH_RUN_NAME") == "simulation_manual"


def test_main_calls_batch_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOOKRECS_BATCH_EXECUTION_DATE", "2026-04-03")

    called: dict[str, int] = {"batch": 0}

    def _fake_batch() -> None:
        called["batch"] += 1

    monkeypatch.setattr(mod, "_run_batch", _fake_batch)
    mod.main()

    assert called["batch"] == 1
