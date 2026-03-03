from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FinalItem:
    user_id: Any
    item_id: Any
    source: str
    final_score: float
    features: dict[str, float] = field(default_factory=dict)
