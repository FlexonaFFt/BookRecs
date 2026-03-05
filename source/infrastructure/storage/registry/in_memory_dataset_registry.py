from __future__ import annotations

from source.application.ports import DatasetRegistryPort
from source.domain.entities import DatasetVersion
# Хранит версии датасета в памяти для локального выполнения.
class InMemoryDatasetRegistry(DatasetRegistryPort):
    """In-memory registry for local runs."""

    def __init__(self) -> None:
        self._by_key: dict[tuple[str, str], DatasetVersion] = {}

    def find_success_by_hash(self, dataset_name: str, params_hash: str) -> DatasetVersion | None:
        return self._by_key.get((dataset_name, params_hash))

    def upsert(self, dataset_version: DatasetVersion) -> None:
        self._by_key[(dataset_version.dataset_name, dataset_version.params_hash)] = dataset_version
