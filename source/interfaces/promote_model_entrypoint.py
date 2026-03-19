from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from source.infrastructure.inference import build_local_pointer, write_model_pointer


def _env_str(name: str, default: str = "") -> str:
    return str(os.getenv(name, default)).strip()


def _env_bool(name: str, default: bool) -> bool:
    value = _env_str(name)
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_optional_float(name: str) -> float | None:
    value = _env_str(name)
    if not value:
        return None
    return float(value)


def _resolve_run_id() -> str:
    run_id = _env_str("BOOKRECS_PROMOTE_RUN_NAME") or _env_str(
        "BOOKRECS_TRAIN_RUN_NAME"
    )
    if not run_id:
        raise ValueError(
            "BOOKRECS_PROMOTE_RUN_NAME (or BOOKRECS_TRAIN_RUN_NAME) is required"
        )
    return run_id


def _load_manifest(output_root: str, run_id: str) -> dict[str, Any]:
    manifest_path = Path(output_root) / run_id / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _validate_manifest(manifest: dict[str, Any], *, require_success: bool) -> None:
    if require_success:
        status = str(manifest.get("status", "")).upper()
        if status != "SUCCESS":
            raise ValueError(f"run status is not SUCCESS: {status or 'UNKNOWN'}")


def _validate_thresholds(manifest: dict[str, Any]) -> None:
    metrics = manifest.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    thresholds = {
        "ndcg@10": {
            "threshold": _env_optional_float("BOOKRECS_PROMOTION_MIN_NDCG10"),
            "aliases": ("ndcg@10", "ndcg_at_k"),
        },
        "recall@10": {
            "threshold": _env_optional_float("BOOKRECS_PROMOTION_MIN_RECALL10"),
            "aliases": ("recall@10", "recall_at_k"),
        },
        "cold_ndcg@10": {
            "threshold": _env_optional_float("BOOKRECS_PROMOTION_MIN_COLD_NDCG10"),
            "aliases": ("cold_ndcg@10", "cold_ndcg_at_k"),
        },
        "cold_recall@10": {
            "threshold": _env_optional_float("BOOKRECS_PROMOTION_MIN_COLD_RECALL10"),
            "aliases": ("cold_recall@10", "cold_recall_at_k"),
        },
    }

    failed: list[str] = []
    for key, spec in thresholds.items():
        threshold_val: float | None = spec["threshold"]  # type: ignore[assignment]
        if threshold_val is None:
            continue
        aliases: tuple[str, ...] = spec["aliases"]  # type: ignore[assignment]
        value = None
        for alias in aliases:
            if alias in metrics:
                value = metrics.get(alias)
                break
        if value is None:
            failed.append(f"{key}=MISSING < {threshold_val}")
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            failed.append(f"{key}=INVALID < {threshold_val}")
            continue
        if numeric < threshold_val:
            failed.append(f"{key}={numeric:.6f} < {threshold_val}")

    if failed:
        raise ValueError("promotion thresholds not met: " + "; ".join(failed))


def main() -> None:
    output_root = _env_str("BOOKRECS_TRAIN_OUTPUT_ROOT", "artifacts/runs")
    pointer_path = _env_str(
        "BOOKRECS_ACTIVE_MODEL_POINTER", "artifacts/runs/active_model.json"
    )
    require_success = _env_bool("BOOKRECS_PROMOTION_REQUIRE_SUCCESS", True)
    run_id = _resolve_run_id()

    manifest = _load_manifest(output_root=output_root, run_id=run_id)
    _validate_manifest(manifest, require_success=require_success)
    _validate_thresholds(manifest)

    pointer = build_local_pointer(
        run_id=run_id,
        output_root=output_root,
        metrics=(
            manifest.get("metrics", {})
            if isinstance(manifest.get("metrics"), dict)
            else {}
        ),
    )
    write_model_pointer(pointer_path, pointer)
    print(
        f"[promote] success: run_id={run_id} "
        f"model_uri={pointer.model_uri} "
        f"pointer={pointer_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
