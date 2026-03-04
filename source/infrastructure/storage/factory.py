from __future__ import annotations

from dataclasses import dataclass

from source.application.ports import DatasetRegistryPort, DatasetStorePort, RunLogPort
from source.infrastructure.storage.postgres import ClientPg
from source.infrastructure.storage.registry import RegistryMemory, RegistryPg
from source.infrastructure.storage.runlog import RunLogMemory, RunLogPg
from source.infrastructure.storage.store import StoreLocal, StoreS3


@dataclass(frozen=True)
class PrepareStorageBackends:
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
) -> PrepareStorageBackends:
    if registry_backend == "postgres":
        if not pg_dsn.strip():
            raise ValueError("pg-dsn is required when --registry-backend=postgres")
        pg = ClientPg(pg_dsn)
        dataset_registry: DatasetRegistryPort = RegistryPg(pg)
        run_log: RunLogPort = RunLogPg(pg)
    else:
        dataset_registry = RegistryMemory()
        run_log = RunLogMemory()

    if store_backend == "s3":
        dataset_store: DatasetStorePort = StoreS3(
            bucket=s3_bucket,
            region=s3_region,
            endpoint_url=s3_endpoint,
        )
    else:
        dataset_store = StoreLocal()

    return PrepareStorageBackends(
        dataset_registry=dataset_registry,
        run_log=run_log,
        dataset_store=dataset_store,
    )
