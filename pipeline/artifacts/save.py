from __future__ import annotations

import json
import pickle
import pandas as pd

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from src.data_bundle import DataBundle


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def prepare_run_dir(base_dir: str = "pipeline/output",
                    run_name: Optional[str] = None) -> Path:

    root = Path(base_dir)
    name = run_name or _utc_stamp()
    run_dir = root / name
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_pickle(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def save_json(data: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def save_metrics(metrics_df: pd.DataFrame, run_dir: Path) -> dict[str, str]:
    reports_dir = run_dir / "reports"
    csv_path = reports_dir / "metrics.csv"
    json_path = reports_dir / "metrics.json"
    metrics_df.to_csv(csv_path, index=False)
    metrics_df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    return {"metrics_csv": str(csv_path), "metrics_json": str(json_path)}


def save_models(models: dict[str, Any], run_dir: Path) -> dict[str, str]:
    model_dir = run_dir / "models"
    paths: dict[str, str] = {}
    for name, model in models.items():
        path = model_dir / f"{name}.pkl"
        save_pickle(model, path)
        paths[name] = str(path)
    return paths


def bundle_metadata(bundle: DataBundle) -> dict[str, Any]:
    return {
        "data_root": bundle.data_root,
        "split_name": bundle.split_name,
        "catalog_size": int(bundle.catalog_size),
        "eval_users": int(len(bundle.eval_users)),
        "warm_items": int(len(bundle.warm_item_ids)),
        "cold_items": int(len(bundle.cold_item_ids)),
        "summary": bundle.summary,
    }


def save_run_manifest(run_dir: Path, *, config: dict[str, Any],
    bundle: DataBundle, model_paths: dict[str, str], report_paths: dict[str, str]) -> str:

    manifest = {
        "run_dir": str(run_dir),
        "created_at_utc": _utc_stamp(),
        "config": config,
        "bundle": bundle_metadata(bundle),
        "models": model_paths,
        "reports": report_paths,
    }
    save_json(manifest, run_dir / "run_manifest.json")
    return str(run_dir / "run_manifest.json")


def save_training_outputs(*, metrics_df: pd.DataFrame, models: dict[str, Any],
    config: dict[str, Any], bundle: DataBundle, base_dir: str = "pipeline/output",
    run_name: Optional[str] = None) -> dict[str, Any]:

    run_dir = prepare_run_dir(base_dir=base_dir, run_name=run_name)
    report_paths = save_metrics(metrics_df, run_dir)
    model_paths = save_models(models, run_dir)
    manifest_path = save_run_manifest(
        run_dir,
        config=config,
        bundle=bundle,
        model_paths=model_paths,
        report_paths=report_paths,
    )
    return {
        "run_dir": str(run_dir),
        "manifest": manifest_path,
        "models": model_paths,
        "reports": report_paths,
    }
