from __future__ import annotations

from dataclasses import dataclass

from source.application.ports import DatasetRegistryPort, DatasetStorePort, RunLogPort
from source.infrastructure.storage.postgres import PostgresClient
from source.infrastructure.storage.registry.in_memory_dataset_registry import (
    InMemoryDatasetRegistry,
)
from source.infrastructure.storage.registry.postgres_dataset_registry import (
    PostgresDatasetRegistry,
)
from source.infrastructure.storage.runlog.in_memory_run_log import InMemoryRunLog
from source.infrastructure.storage.runlog.postgres_run_log import PostgresRunLog
from source.infrastructure.storage.store.local_dataset_store import LocalDatasetStore
from source.infrastructure.storage.store.s3_dataset_store import S3DatasetStore


@dataclass(frozen=True)
# Объединяет конкретные бэкенды хранилища для пайплайна подготовки данных.
class PrepareDataStorageBackends:
    dataset_registry: DatasetRegistryPort
    run_log: RunLogPort
    dataset_store: DatasetStorePort


def build_prepare_storage_backends(
    *,
    registry_backend: str,
    pg_dsn: str,
    store_backend: str,
    s3_bucket: str,
    s3_region: str,
    s3_endpoint: str,
    s3_verify_ssl: bool = True,
) -> PrepareDataStorageBackends:
    if registry_backend == "postgres":
        if not pg_dsn.strip():
            raise ValueError("pg-dsn is required when --registry-backend=postgres")
        pg = PostgresClient(pg_dsn)
        dataset_registry: DatasetRegistryPort = PostgresDatasetRegistry(pg)
        run_log: RunLogPort = PostgresRunLog(pg)
    else:
        dataset_registry = InMemoryDatasetRegistry()
        run_log = InMemoryRunLog()

    if store_backend == "s3":
        dataset_store: DatasetStorePort = S3DatasetStore(
            bucket=s3_bucket,
            region=s3_region,
            endpoint_url=s3_endpoint,
            verify_ssl=s3_verify_ssl,
        )
    else:
        dataset_store = LocalDatasetStore()

    return PrepareDataStorageBackends(
        dataset_registry=dataset_registry,
        run_log=run_log,
        dataset_store=dataset_store,
    )
