from __future__ import annotations

import json
from pathlib import Path

from source.infrastructure.inference.model_registry import (
    build_local_pointer,
    read_model_pointer,
    resolve_model_uri,
    write_model_pointer,
)


def test_pointer_read_write_roundtrip(tmp_path: Path) -> None:
    pointer = build_local_pointer(run_id="batch_20260310", output_root=str(tmp_path), metrics={"ndcg_at_k": 0.4})
    path = tmp_path / "active_model.json"
    write_model_pointer(str(path), pointer)
    loaded = read_model_pointer(str(path))
    assert loaded is not None
    assert loaded.run_id == "batch_20260310"
    assert loaded.metrics["ndcg_at_k"] == 0.4


def test_resolve_model_uri_prefers_pointer(tmp_path: Path) -> None:
    pointer_path = tmp_path / "active_model.json"
    payload = {
        "run_id": "batch_20260310",
        "model_uri": "/tmp/modelA",
        "promoted_at": "2026-03-10T00:00:00+00:00",
        "metrics": {},
    }
    pointer_path.write_text(json.dumps(payload), encoding="utf-8")
    uri, pointer = resolve_model_uri("/tmp/default", str(pointer_path))
    assert uri == "/tmp/modelA"
    assert pointer is not None
    assert pointer.run_id == "batch_20260310"


def test_resolve_model_uri_falls_back_to_config(tmp_path: Path) -> None:
    uri, pointer = resolve_model_uri("/tmp/default", str(tmp_path / "missing.json"))
    assert uri == "/tmp/default"
    assert pointer is None
