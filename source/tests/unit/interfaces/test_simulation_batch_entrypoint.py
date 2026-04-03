from __future__ import annotations

import os
from datetime import date

import pytest

from source.interfaces import simulation_batch_entrypoint as mod


def test_compute_cutoff_at_window_start() -> None:
    result = mod._compute_cutoff_date(
        execution_date=date(2026, 3, 1),
        exec_window_start=date(2026, 3, 1),
        exec_window_end=date(2026, 4, 1),
        data_start=date(2011, 1, 1),
        data_end=date(2017, 1, 1),
    )
    assert result == date(2011, 1, 1)


def test_compute_cutoff_at_window_end() -> None:
    result = mod._compute_cutoff_date(
        execution_date=date(2026, 4, 1),
        exec_window_start=date(2026, 3, 1),
        exec_window_end=date(2026, 4, 1),
        data_start=date(2011, 1, 1),
        data_end=date(2017, 1, 1),
    )
    assert result == date(2017, 1, 1)


def test_compute_cutoff_midpoint() -> None:
    result = mod._compute_cutoff_date(
        execution_date=date(2026, 3, 16),
        exec_window_start=date(2026, 3, 1),
        exec_window_end=date(2026, 4, 1),
        data_start=date(2011, 1, 1),
        data_end=date(2013, 1, 1),
    )
    assert result > date(2011, 1, 1)
    assert result < date(2013, 1, 1)


def test_compute_cutoff_clamps_before_window_start() -> None:
    result = mod._compute_cutoff_date(
        execution_date=date(2026, 2, 1),
        exec_window_start=date(2026, 3, 1),
        exec_window_end=date(2026, 4, 1),
        data_start=date(2011, 1, 1),
        data_end=date(2017, 1, 1),
    )
    assert result == date(2011, 1, 1)


def test_compute_cutoff_clamps_after_window_end() -> None:
    result = mod._compute_cutoff_date(
        execution_date=date(2026, 5, 1),
        exec_window_start=date(2026, 3, 1),
        exec_window_end=date(2026, 4, 1),
        data_start=date(2011, 1, 1),
        data_end=date(2017, 1, 1),
    )
    assert result == date(2017, 1, 1)


def test_compute_cutoff_zero_window_returns_data_end() -> None:
    result = mod._compute_cutoff_date(
        execution_date=date(2026, 3, 15),
        exec_window_start=date(2026, 3, 1),
        exec_window_end=date(2026, 3, 1),
        data_start=date(2011, 1, 1),
        data_end=date(2017, 1, 1),
    )
    assert result == date(2017, 1, 1)


def test_main_sets_env_and_calls_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOOKRECS_BATCH_EXECUTION_DATE", "2026-03-16")
    monkeypatch.setenv("BOOKRECS_SIMULATION_WINDOW_START", "2026-03-01")
    monkeypatch.setenv("BOOKRECS_SIMULATION_WINDOW_END", "2026-04-01")
    monkeypatch.setenv("BOOKRECS_SIMULATION_DATA_START", "2011-01-01")
    monkeypatch.setenv("BOOKRECS_SIMULATION_DATA_END", "2017-01-01")
    monkeypatch.delenv("BOOKRECS_BATCH_RUN_NAME", raising=False)

    called: dict[str, int] = {"batch": 0}

    def _fake_batch() -> None:
        called["batch"] += 1

    monkeypatch.setattr(mod, "_run_batch", _fake_batch)
    mod.main()

    assert called["batch"] == 1
    assert os.environ.get("BOOKRECS_BATCH_RUN_NAME") == "simulation_20260316"
    assert os.environ.get("BOOKRECS_SIMULATION_CUTOFF_DATE") is not None
    dataset_name = os.environ.get("BOOKRECS_DATASET_NAME", "")
    assert dataset_name.startswith("goodreads_ya_sim_")
    assert os.environ.get("BOOKRECS_TRAIN_DATASET_DIR") == (
        f"artifacts/tmp_preprocessed/{dataset_name}"
    )


def test_main_uses_data_end_when_no_execution_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BOOKRECS_BATCH_EXECUTION_DATE", raising=False)
    monkeypatch.setenv("BOOKRECS_SIMULATION_DATA_END", "2017-01-01")

    called: dict[str, int] = {"batch": 0}

    def _fake_batch() -> None:
        called["batch"] += 1

    monkeypatch.setattr(mod, "_run_batch", _fake_batch)
    mod.main()

    assert called["batch"] == 1
    assert os.environ.get("BOOKRECS_SIMULATION_CUTOFF_DATE") == "2017-01-01"
    assert os.environ.get("BOOKRECS_BATCH_RUN_NAME") == "simulation_manual"
