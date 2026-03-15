from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
# Описывает кандидата.
class Candidate:
    user_id: Any
    item_id: Any
    source: str
    score: float
    features: dict[str, float] = field(default_factory=dict)
