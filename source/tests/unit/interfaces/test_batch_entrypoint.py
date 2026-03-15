from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from source.interfaces import batch_entrypoint as mod


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2026-03-09", "20260309"),
        ("2026-03-09T12:13:14+00:00", "20260309"),
        ("  ", "manual"),
        (None, "manual"),
    ],
)
def test_normalize_date_key(raw: str | None, expected: str) -> None:
    assert mod._normalize_date_key(raw) == expected


def test_build_run_name_prefers_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOOKRECS_BATCH_RUN_NAME", "my_custom_run")
    monkeypatch.setenv("BOOKRECS_BATCH_EXECUTION_DATE", "2026-03-09")
    assert mod._build_run_name() == "my_custom_run"


def test_build_run_name_uses_execution_date(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOOKRECS_BATCH_RUN_NAME", raising=False)
    monkeypatch.setenv("BOOKRECS_BATCH_EXECUTION_DATE", "2026-03-09")
    assert mod._build_run_name() == "batch_20260309"


def test_success_manifest_exists_true_for_success(tmp_path: Path) -> None:
    run_dir = tmp_path / "batch_20260309"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "SUCCESS"}),
        encoding="utf-8",
    )
    assert mod._success_manifest_exists(str(tmp_path), "batch_20260309") is True


def test_success_manifest_exists_false_for_non_success(tmp_path: Path) -> None:
    run_dir = tmp_path / "batch_20260309"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "FAILED"}),
        encoding="utf-8",
    )
    assert mod._success_manifest_exists(str(tmp_path), "batch_20260309") is False


def test_main_skips_when_success_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_name = "batch_20260309"
    run_dir = tmp_path / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"status": "SUCCESS"}), encoding="utf-8")

    monkeypatch.setenv("BOOKRECS_BATCH_EXECUTION_DATE", "2026-03-09")
    monkeypatch.delenv("BOOKRECS_BATCH_RUN_NAME", raising=False)

    called = {"pipeline": 0}

    def _fake_run_pipeline() -> None:
        called["pipeline"] += 1

    monkeypatch.setattr(mod, "run_pipeline_from_env", _fake_run_pipeline)
    monkeypatch.setattr(mod, "load_pipeline_settings", lambda: SimpleNamespace(output_root=str(tmp_path)))

    mod.main()

    assert called["pipeline"] == 0
    assert mod.os.environ.get("BOOKRECS_TRAIN_RUN_NAME") == run_name


def test_main_runs_when_no_success_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_name = "batch_20260310"

    monkeypatch.setenv("BOOKRECS_BATCH_EXECUTION_DATE", "2026-03-10")
    monkeypatch.delenv("BOOKRECS_BATCH_RUN_NAME", raising=False)

    called = {"pipeline": 0}

    def _fake_run_pipeline() -> None:
        called["pipeline"] += 1

    monkeypatch.setattr(mod, "run_pipeline_from_env", _fake_run_pipeline)
    monkeypatch.setattr(mod, "load_pipeline_settings", lambda: SimpleNamespace(output_root=str(tmp_path)))

    mod.main()

    assert called["pipeline"] == 1
    assert mod.os.environ.get("BOOKRECS_TRAIN_RUN_NAME") == run_name
