from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Candidate:
    user_id: Any
    item_id: Any
    source: str
    score: float
