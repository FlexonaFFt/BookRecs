from __future__ import annotations

from abc import ABC, abstractmethod

from source.domain.entities import DatasetVersion


class DatasetRegistryPort(ABC):
    """
    Реестр версий датасета (целевая реализация: PostgreSQL).
    """

    @abstractmethod
    def find_success_by_hash(self, dataset_name: str, params_hash: str) -> DatasetVersion | None:
        """
        Возвращает успешную версию датасета с тем же params_hash, если она есть.
        """
        raise NotImplementedError

    @abstractmethod
    def upsert(self, dataset_version: DatasetVersion) -> None:
        """
        Создает или обновляет запись версии датасета.
        """
        raise NotImplementedError
