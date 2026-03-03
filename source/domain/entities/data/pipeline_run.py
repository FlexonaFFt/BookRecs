from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from source.domain.entities.data.run_status import RunStatus


@dataclass
class PipelineRun:
    run_id: str
    pipeline_name: str
    status: RunStatus = RunStatus.STARTED
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    message: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def mark_success(self, message: str = "") -> None:
        self.status = RunStatus.SUCCESS
        self.finished_at = datetime.now(timezone.utc)
        self.message = message

    def mark_failed(self, message: str) -> None:
        self.status = RunStatus.FAILED
        self.finished_at = datetime.now(timezone.utc)
        self.message = message

    def mark_skipped(self, message: str) -> None:
        self.status = RunStatus.SKIPPED
        self.finished_at = datetime.now(timezone.utc)
        self.message = message
