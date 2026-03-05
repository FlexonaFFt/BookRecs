from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import boto3
except ModuleNotFoundError:
    boto3 = None


@dataclass(frozen=True)
class ModelBundle:
    stage1: dict[str, Any]
    stage2: Any
    stage3: Any
    model_dir: str
    metrics_snapshot: dict[str, Any]


class ModelBundleLoader:
    def __init__(
        self,
        *,
        s3_region: str,
        s3_endpoint: str,
        local_cache_root: str = "artifacts/cache/models",
    ) -> None:
        self._s3_region = s3_region or "us-east-1"
        self._s3_endpoint = s3_endpoint or None
        self._local_cache_root = Path(local_cache_root)
        self._s3_client = None

    def load(self, model_uri: str | None) -> ModelBundle:
        resolved = model_uri.strip() if model_uri else ""
        if not resolved:
            resolved = self._discover_local_model_dir()
        if resolved.startswith("s3://"):
            model_dir = self._download_s3_model_dir(resolved)
        else:
            model_dir = Path(resolved).expanduser().resolve()
        return self._read_bundle(model_dir)

    def _read_bundle(self, model_dir: Path) -> ModelBundle:
        stage1_path = model_dir / "stage1.pkl"
        stage2_path = model_dir / "stage2.pkl"
        stage3_path = model_dir / "stage3.pkl"
        metrics_path = model_dir / "metrics_snapshot.json"
        required = [stage1_path, stage2_path, stage3_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing model artifacts: {missing}")

        with open(stage1_path, "rb") as f:
            stage1 = pickle.load(f)
        with open(stage2_path, "rb") as f:
            stage2 = pickle.load(f)
        with open(stage3_path, "rb") as f:
            stage3 = pickle.load(f)

        metrics_snapshot: dict[str, Any] = {}
        if metrics_path.exists():
            metrics_snapshot = json.loads(metrics_path.read_text(encoding="utf-8"))
        return ModelBundle(
            stage1=stage1,
            stage2=stage2,
            stage3=stage3,
            model_dir=str(model_dir),
            metrics_snapshot=metrics_snapshot,
        )

    def _discover_local_model_dir(self) -> str:
        candidates: list[Path] = []
        for root in [Path("artifacts/runs"), Path("artifacts/baselines")]:
            if not root.exists():
                continue
            for path in root.glob("*/models"):
                if (path / "stage1.pkl").exists():
                    candidates.append(path)
        if not candidates:
            raise FileNotFoundError(
                "No local model directory found. Set BOOKRECS_API_MODEL_URI to local path or s3:// prefix."
            )
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(candidates[0].resolve())

    def _download_s3_model_dir(self, uri: str) -> Path:
        parsed = urlparse(uri)
        bucket = parsed.netloc
        prefix = parsed.path.strip("/")
        if not bucket:
            raise ValueError(f"Invalid model s3 URI: {uri}")
        cache_dir = self._local_cache_root / bucket / prefix
        cache_dir.mkdir(parents=True, exist_ok=True)

        for name in ["stage1.pkl", "stage2.pkl", "stage3.pkl", "metrics_snapshot.json"]:
            key = f"{prefix}/{name}" if prefix else name
            dst = cache_dir / name
            self._s3().download_file(bucket, key, str(dst))
        return cache_dir

    def _s3(self):
        if boto3 is None:
            raise RuntimeError("boto3 is required for S3 model loading. Install dependency: pip install boto3")
        if self._s3_client is None:
            self._s3_client = boto3.client(
                "s3",
                region_name=self._s3_region,
                endpoint_url=self._s3_endpoint,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )
        return self._s3_client
