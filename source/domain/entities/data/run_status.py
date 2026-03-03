from __future__ import annotations

from enum import Enum


class RunStatus(str, Enum):
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
