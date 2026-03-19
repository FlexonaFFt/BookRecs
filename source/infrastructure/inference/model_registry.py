from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelPointer:
    run_id: str
    model_uri: str
    promoted_at: str
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_uri": self.model_uri,
            "promoted_at": self.promoted_at,
            "metrics": self.metrics,
        }


def read_model_pointer(pointer_path: str | None) -> ModelPointer | None:
    path = _safe_pointer_path(pointer_path)
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    run_id = str(payload.get("run_id", "")).strip()
    model_uri = str(payload.get("model_uri", "")).strip()
    promoted_at = str(payload.get("promoted_at", "")).strip()
    metrics = payload.get("metrics", {})
    if not run_id or not model_uri:
        return None
    if not isinstance(metrics, dict):
        metrics = {}
    if not promoted_at:
        promoted_at = datetime.now(timezone.utc).isoformat()
    return ModelPointer(
        run_id=run_id,
        model_uri=model_uri,
        promoted_at=promoted_at,
        metrics=metrics,
    )


def write_model_pointer(pointer_path: str, pointer: ModelPointer) -> None:
    path = _safe_pointer_path(pointer_path)
    if path is None:
        raise ValueError("pointer_path is required")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(pointer.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def resolve_model_uri(
    config_model_uri: str, pointer_path: str | None
) -> tuple[str, ModelPointer | None]:
    pointer = read_model_pointer(pointer_path)
    if pointer is not None:
        return pointer.model_uri, pointer
    return config_model_uri.strip(), None


def build_local_pointer(
    run_id: str, output_root: str, metrics: dict[str, Any] | None = None
) -> ModelPointer:
    model_uri = str((Path(output_root) / run_id / "models").resolve())
    return ModelPointer(
        run_id=run_id,
        model_uri=model_uri,
        promoted_at=datetime.now(timezone.utc).isoformat(),
        metrics=metrics or {},
    )


def _safe_pointer_path(pointer_path: str | None) -> Path | None:
    value = (pointer_path or "").strip()
    if not value:
        return None
    return Path(value).expanduser().resolve()
