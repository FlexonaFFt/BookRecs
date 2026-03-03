from __future__ import annotations

from abc import ABC, abstractmethod

from source.domain.entities import DatasetArtifacts, DatasetVersion


class DatasetStorePort(ABC):
    """
    Хранилище файловых артефактов датасета (целевая реализация: S3).
    """

    @abstractmethod
    def save(self, dataset_version: DatasetVersion, artifacts: DatasetArtifacts) -> DatasetArtifacts:
        """
        Сохраняет артефакты в постоянное хранилище.

        Возвращает DatasetArtifacts с уже финальными URI (обычно s3://...).
        """
        raise NotImplementedError

    @abstractmethod
    def exists(self, dataset_version: DatasetVersion) -> bool:
        """
        Проверяет, существует ли уже собранная версия датасета.
        """
        raise NotImplementedError
