from __future__ import annotations

from enum import Enum


# Описывает возможные статусы запуска.
class RunStatus(str, Enum):
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
