from __future__ import annotations

from datetime import date

import pytest

from source.interfaces import batch_backfill_entrypoint as mod


def test_parse_days_default() -> None:
    assert mod._parse_days(None) == 5
    assert mod._parse_days("") == 5


def test_parse_days_valid() -> None:
    assert mod._parse_days("7") == 7


def test_parse_days_invalid() -> None:
    with pytest.raises(ValueError):
        mod._parse_days("0")


def test_parse_end_date_default(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Date(date):
        @classmethod
        def today(cls) -> _Date:
            return cls(2026, 3, 10)

    monkeypatch.setattr(mod, "date", _Date)
    assert mod._parse_end_date(None).isoformat() == "2026-03-10"


def test_parse_end_date_invalid() -> None:
    with pytest.raises(ValueError):
        mod._parse_end_date("10-03-2026")


def test_build_dates() -> None:
    result = mod._build_dates(date(2026, 3, 10), 3)
    assert result == ["2026-03-08", "2026-03-09", "2026-03-10"]


def test_parse_promote_enabled() -> None:
    assert mod._parse_promote_enabled(None) is True
    assert mod._parse_promote_enabled("") is True
    assert mod._parse_promote_enabled("true") is True
    assert mod._parse_promote_enabled("1") is True
    assert mod._parse_promote_enabled("false") is False


def test_main_runs_over_expected_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOOKRECS_BATCH_BACKFILL_DAYS", "3")
    monkeypatch.setenv("BOOKRECS_BATCH_END_DATE", "2026-03-10")
    monkeypatch.setenv("BOOKRECS_BATCH_RUN_NAME", "must_be_removed")
    monkeypatch.setenv("BOOKRECS_BATCH_BACKFILL_PROMOTE", "true")

    calls: list[str] = []
    promotions: list[str] = []

    def _fake_run_batch_once() -> None:
        calls.append(mod.os.environ["BOOKRECS_BATCH_EXECUTION_DATE"])

    def _fake_promote() -> None:
        promotions.append(mod.os.environ["BOOKRECS_PROMOTE_RUN_NAME"])

    monkeypatch.setattr(mod, "run_batch_once", _fake_run_batch_once)
    monkeypatch.setattr(mod, "promote_model", _fake_promote)

    mod.main()

    assert calls == ["2026-03-08", "2026-03-09", "2026-03-10"]
    assert promotions == ["batch_20260308", "batch_20260309", "batch_20260310"]
    assert "BOOKRECS_BATCH_RUN_NAME" not in mod.os.environ
