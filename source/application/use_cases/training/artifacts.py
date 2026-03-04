from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ARTIFACT_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
# Описывает структуру хранения артефактов.
class ArtifactLayout:
    run_dir: Path
    model_dir: Path
    stage1_model: Path
    stage2_config: Path
    stage3_config: Path
    metrics_snapshot: Path
    metrics: Path
    timings: Path
    manifest: Path
    train_log: Path

    def to_relative_map(self) -> dict[str, str]:
        return {
            "models_dir": self.model_dir.relative_to(self.run_dir).as_posix(),
            "stage1_model": self.stage1_model.relative_to(self.run_dir).as_posix(),
            "stage2_config": self.stage2_config.relative_to(self.run_dir).as_posix(),
            "stage3_config": self.stage3_config.relative_to(self.run_dir).as_posix(),
            "metrics_snapshot": self.metrics_snapshot.relative_to(self.run_dir).as_posix(),
            "metrics": self.metrics.relative_to(self.run_dir).as_posix(),
            "timings": self.timings.relative_to(self.run_dir).as_posix(),
            "manifest": self.manifest.relative_to(self.run_dir).as_posix(),
            "train_log": self.train_log.relative_to(self.run_dir).as_posix(),
        }


def build_layout(run_dir: Path) -> ArtifactLayout:
    model_dir = run_dir / "models"
    return ArtifactLayout(
        run_dir=run_dir,
        model_dir=model_dir,
        stage1_model=model_dir / "stage1.pkl",
        stage2_config=model_dir / "stage2.json",
        stage3_config=model_dir / "stage3.json",
        metrics_snapshot=model_dir / "metrics_snapshot.json",
        metrics=run_dir / "metrics.json",
        timings=run_dir / "timings.json",
        manifest=run_dir / "manifest.json",
        train_log=run_dir / "train.log.jsonl",
    )


@dataclass(frozen=True)
# Описывает манифест обучения.
class TrainManifest:
    schema_version: str
    run_id: str
    status: str
    duration_sec: float
    dataset_dir: str
    config: dict[str, Any]
    config_hash: str
    artifacts: dict[str, str]
    metrics: dict[str, float]
    timings: dict[str, float]

    @staticmethod
    def build(
        *,
        run_id: str,
        status: str,
        duration_sec: float,
        dataset_dir: str,
        config: dict[str, Any],
        layout: ArtifactLayout,
        metrics: dict[str, float],
        timings: dict[str, float],
    ) -> "TrainManifest":
        return TrainManifest(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            run_id=run_id,
            status=status,
            duration_sec=duration_sec,
            dataset_dir=dataset_dir,
            config=config,
            config_hash=_stable_hash(config),
            artifacts=layout.to_relative_map(),
            metrics=metrics,
            timings=timings,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stable_hash(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
