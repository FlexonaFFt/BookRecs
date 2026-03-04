from __future__ import annotations

from abc import ABC, abstractmethod

from source.domain.entities import (
    DatasetArtifacts,
    DatasetSource,
    DatasetVersion,
    PipelineRun,
    PreprocessingParams,
)


class PreprocessorPort(ABC):
    @abstractmethod
    def run(self, source: DatasetSource, params: PreprocessingParams) -> DatasetArtifacts:
        raise NotImplementedError


class DatasetStorePort(ABC):
    @abstractmethod
    def save(self, dataset_version: DatasetVersion, artifacts: DatasetArtifacts) -> DatasetArtifacts:
        raise NotImplementedError

    @abstractmethod
    def exists(self, dataset_version: DatasetVersion) -> bool:
        raise NotImplementedError


class DatasetRegistryPort(ABC):
    @abstractmethod
    def find_success_by_hash(self, dataset_name: str, params_hash: str) -> DatasetVersion | None:
        raise NotImplementedError

    @abstractmethod
    def upsert(self, dataset_version: DatasetVersion) -> None:
        raise NotImplementedError


class RunLogPort(ABC):
    @abstractmethod
    def start(self, run: PipelineRun) -> None:
        raise NotImplementedError

    @abstractmethod
    def finish(self, run: PipelineRun) -> None:
        raise NotImplementedError
