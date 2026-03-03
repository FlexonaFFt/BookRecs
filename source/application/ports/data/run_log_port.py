from __future__ import annotations

from abc import ABC, abstractmethod

from source.domain.entities import PipelineRun


class RunLogPort(ABC):
    """
    Логирование технического статуса пайплайна (целевая реализация: PostgreSQL).
    """

    @abstractmethod
    def start(self, run: PipelineRun) -> None:
        raise NotImplementedError

    @abstractmethod
    def finish(self, run: PipelineRun) -> None:
        raise NotImplementedError
