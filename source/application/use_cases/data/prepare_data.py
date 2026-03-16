from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any
from uuid import uuid4

from source.application.ports import (
    DatasetRegistryPort,
    DatasetStorePort,
    PreprocessorPort,
    RunLogPort,
)
from source.domain.entities import (
    DatasetSource,
    DatasetVersion,
    PipelineRun,
    PreprocessingParams,
)


@dataclass(frozen=True)
# Содержит входные данные команды подготовки данных.
class PrepareDataCommand:
    """Input for dataset preparation flow."""
    dataset_name: str
    source: DatasetSource
    params: PreprocessingParams
    s3_prefix: str
    metadata: dict[str, Any] | None = None
# Реализует сценарий подготовки данных.
class PrepareDataUseCase:
    """Run full prepare-data flow: preprocess -> store -> registry -> runlog."""

    def __init__(
        self,
        preprocessor: PreprocessorPort,
        dataset_store: DatasetStorePort,
        dataset_registry: DatasetRegistryPort,
        run_log: RunLogPort,
    ) -> None:
        self._preprocessor = preprocessor
        self._dataset_store = dataset_store
        self._dataset_registry = dataset_registry
        self._run_log = run_log

    def execute(self, cmd: PrepareDataCommand) -> DatasetVersion:
        cmd.source.validate()
        cmd.params.validate()
        if not cmd.dataset_name.strip():
            raise ValueError("dataset_name is required")
        if not cmd.s3_prefix.strip():
            raise ValueError("s3_prefix is required")

        run = PipelineRun(
            run_id=str(uuid4()),
            pipeline_name="prepare-data",
            metadata={"dataset_name": cmd.dataset_name},
        )
        self._run_log.start(run)
        print(f"[prepare] Старт подготовки dataset={cmd.dataset_name}, run_id={run.run_id}", flush=True)

        try:
            params_hash = self._build_params_hash(cmd)
            print(f"[prepare] params_hash={params_hash}", flush=True)
            existing = self._dataset_registry.find_success_by_hash(
                dataset_name=cmd.dataset_name,
                params_hash=params_hash,
            )

            if existing is not None and self._dataset_store.exists(existing):
                print("[prepare] Найден готовый датасет с тем же params_hash, шаг подготовки пропущен.", flush=True)
                run.mark_skipped("Dataset already exists for the same params hash.")
                self._run_log.finish(run)
                return existing

            print("[prepare] Запуск preprocessor.run(...)", flush=True)
            local_artifacts = self._preprocessor.run(cmd.source, cmd.params)
            print("[prepare] Локальные артефакты готовы, начинаем публикацию в storage backend.", flush=True)

            dataset_version = DatasetVersion(
                dataset_name=cmd.dataset_name,
                version_id=str(uuid4()),
                params_hash=params_hash,
                s3_prefix=cmd.s3_prefix,
                params=cmd.params,
                metadata=cmd.metadata or {},
            )
            dataset_version.validate()

            remote_artifacts = self._dataset_store.save(dataset_version, local_artifacts)
            print("[prepare] Публикация артефактов завершена, обновляем registry.", flush=True)
            dataset_version = DatasetVersion(
                dataset_name=dataset_version.dataset_name,
                version_id=dataset_version.version_id,
                params_hash=dataset_version.params_hash,
                s3_prefix=dataset_version.s3_prefix,
                params=dataset_version.params,
                created_at=dataset_version.created_at,
                stats=dataset_version.stats,
                metadata={
                    **dataset_version.metadata,
                    "artifacts": {
                        "books_uri": remote_artifacts.books_uri,
                        "train_uri": remote_artifacts.train_uri,
                        "test_uri": remote_artifacts.test_uri,
                        "local_train_uri": remote_artifacts.local_train_uri,
                        "local_val_uri": remote_artifacts.local_val_uri,
                        "local_val_warm_uri": remote_artifacts.local_val_warm_uri,
                        "local_val_cold_uri": remote_artifacts.local_val_cold_uri,
                        "summary_uri": remote_artifacts.summary_uri,
                        "manifest_uri": remote_artifacts.manifest_uri,
                    },
                },
            )

            self._dataset_registry.upsert(dataset_version)
            print(
                f"[prepare] DatasetVersion сохранен: version_id={dataset_version.version_id}, "
                f"s3_prefix={dataset_version.s3_prefix}",
                flush=True,
            )

            run.mark_success("Dataset prepared and published.")
            self._run_log.finish(run)
            print(f"[prepare] Завершено успешно, run_id={run.run_id}", flush=True)
            return dataset_version
        except Exception as exc:
            run.mark_failed(str(exc))
            self._run_log.finish(run)
            print(f"[prepare] Ошибка: {exc}", flush=True)
            raise

    @staticmethod
    def _build_params_hash(cmd: PrepareDataCommand) -> str:
        payload = {
            "dataset_name": cmd.dataset_name,
            "books_raw_uri": cmd.source.books_raw_uri,
            "interactions_raw_uri": cmd.source.interactions_raw_uri,
            "params": {
                "k_core": cmd.params.k_core,
                "keep_recent_fraction": cmd.params.keep_recent_fraction,
                "test_fraction": cmd.params.test_fraction,
                "local_val_fraction": cmd.params.local_val_fraction,
                "cold_max_interactions": cmd.params.cold_max_interactions,
                "warm_users_only": cmd.params.warm_users_only,
                "language_filter_enabled": cmd.params.language_filter_enabled,
                "interactions_chunksize": cmd.params.interactions_chunksize,
            },
        }
        return sha256(repr(payload).encode("utf-8")).hexdigest()
