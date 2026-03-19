from __future__ import annotations

import json
from pathlib import Path

import pytest

from source.interfaces import promote_model_entrypoint as mod


def _write_manifest(
    tmp_path: Path, run_id: str, status: str = "SUCCESS", metrics: dict | None = None
) -> None:
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True)
    payload = {
        "run_id": run_id,
        "status": status,
        "metrics": metrics or {},
    }
    (run_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_promote_writes_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_manifest(tmp_path, "batch_20260310", metrics={"ndcg@10": 0.42})
    pointer_path = tmp_path / "active_model.json"

    monkeypatch.setenv("BOOKRECS_TRAIN_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("BOOKRECS_PROMOTE_RUN_NAME", "batch_20260310")
    monkeypatch.setenv("BOOKRECS_ACTIVE_MODEL_POINTER", str(pointer_path))

    mod.main()

    payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "batch_20260310"
    assert payload["model_uri"].endswith("/batch_20260310/models")


def test_promote_checks_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_manifest(tmp_path, "batch_20260310", metrics={"ndcg@10": 0.30})
    pointer_path = tmp_path / "active_model.json"

    monkeypatch.setenv("BOOKRECS_TRAIN_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("BOOKRECS_PROMOTE_RUN_NAME", "batch_20260310")
    monkeypatch.setenv("BOOKRECS_ACTIVE_MODEL_POINTER", str(pointer_path))
    monkeypatch.setenv("BOOKRECS_PROMOTION_MIN_NDCG10", "0.35")

    with pytest.raises(ValueError):
        mod.main()


def test_promote_checks_success_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_manifest(
        tmp_path, "batch_20260310", status="FAILED", metrics={"ndcg@10": 0.50}
    )
    pointer_path = tmp_path / "active_model.json"

    monkeypatch.setenv("BOOKRECS_TRAIN_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("BOOKRECS_PROMOTE_RUN_NAME", "batch_20260310")
    monkeypatch.setenv("BOOKRECS_ACTIVE_MODEL_POINTER", str(pointer_path))
    monkeypatch.setenv("BOOKRECS_PROMOTION_REQUIRE_SUCCESS", "true")

    with pytest.raises(ValueError):
        mod.main()


def test_promote_accepts_legacy_metric_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_manifest(tmp_path, "batch_20260310", metrics={"ndcg_at_k": 0.42})
    pointer_path = tmp_path / "active_model.json"

    monkeypatch.setenv("BOOKRECS_TRAIN_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("BOOKRECS_PROMOTE_RUN_NAME", "batch_20260310")
    monkeypatch.setenv("BOOKRECS_ACTIVE_MODEL_POINTER", str(pointer_path))
    monkeypatch.setenv("BOOKRECS_PROMOTION_MIN_NDCG10", "0.35")

    mod.main()

    payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "batch_20260310"
