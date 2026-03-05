from __future__ import annotations

from source.application.ports import RunLogPort
from source.domain.entities import PipelineRun
# Хранит события запуска пайплайна в памяти для локального выполнения.
class InMemoryRunLog(RunLogPort):


    def __init__(self) -> None:
        self.started_runs: list[PipelineRun] = []
        self.finished_runs: list[PipelineRun] = []

    def start(self, run: PipelineRun) -> None:
        self.started_runs.append(run)

    def finish(self, run: PipelineRun) -> None:
        self.finished_runs.append(run)
