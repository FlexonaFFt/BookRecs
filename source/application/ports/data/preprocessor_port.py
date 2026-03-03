from __future__ import annotations

from abc import ABC, abstractmethod

from source.domain.entities import DatasetArtifacts, DatasetSource, PreprocessingParams


class PreprocessorPort(ABC):
    """
    Запускает batch-предобработку raw данных.

    Реализация в infrastructure должна:
    1) прочитать raw source;
    2) сформировать train/test + local split;
    3) вернуть ссылки на локальные/временные артефакты.
    """

    @abstractmethod
    def run(self, source: DatasetSource, params: PreprocessingParams) -> DatasetArtifacts:
        raise NotImplementedError
