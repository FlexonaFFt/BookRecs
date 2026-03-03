from __future__ import annotations

import shutil
from pathlib import Path

from source.application.ports import DatasetStorePort
from source.domain.entities import DatasetArtifacts, DatasetVersion


class StoreLocal(DatasetStorePort):


    def __init__(self, root_dir: str = "artifacts/datasets") -> None:
        self._root_dir = Path(root_dir)

    def save(self, dataset_version: DatasetVersion, artifacts: DatasetArtifacts) -> DatasetArtifacts:
        target_dir = self._target_dir(dataset_version)
        target_dir.mkdir(parents=True, exist_ok=True)

        books_uri = self._copy_to_target(artifacts.books_uri, target_dir / "books.parquet")
        train_uri = self._copy_to_target(artifacts.train_uri, target_dir / "train.parquet")
        test_uri = self._copy_to_target(artifacts.test_uri, target_dir / "test.parquet")
        local_train_uri = self._copy_to_target(artifacts.local_train_uri, target_dir / "local_train.parquet")
        local_val_uri = self._copy_to_target(artifacts.local_val_uri, target_dir / "local_val.parquet")
        local_val_warm_uri = self._copy_to_target(artifacts.local_val_warm_uri, target_dir / "local_val_warm.parquet")
        local_val_cold_uri = self._copy_to_target(artifacts.local_val_cold_uri, target_dir / "local_val_cold.parquet")
        summary_uri = self._copy_to_target(artifacts.summary_uri, target_dir / "summary.json")
        manifest_uri = self._copy_to_target(artifacts.manifest_uri, target_dir / "manifest.json")

        return DatasetArtifacts(
            books_uri=books_uri,
            train_uri=train_uri,
            test_uri=test_uri,
            local_train_uri=local_train_uri,
            local_val_uri=local_val_uri,
            local_val_warm_uri=local_val_warm_uri,
            local_val_cold_uri=local_val_cold_uri,
            summary_uri=summary_uri,
            manifest_uri=manifest_uri,
        )

    def exists(self, dataset_version: DatasetVersion) -> bool:
        manifest = self._target_dir(dataset_version) / "manifest.json"
        return manifest.exists()

    def _target_dir(self, dataset_version: DatasetVersion) -> Path:
        return self._root_dir / dataset_version.dataset_name / dataset_version.version_id

    @staticmethod
    def _copy_to_target(src: str, dst: Path) -> str:
        source = Path(src)
        if not source.exists():
            raise FileNotFoundError(f"Artifact not found: {source}")
        shutil.copy2(source, dst)
        return str(dst)
