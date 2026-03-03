from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from source.domain.entities import Candidate, ScoredCandidate


class PreRankerPort(ABC):
    """Contract for Stage-2 pre-ranking model."""
    @abstractmethod
    def rank(
        self,
        candidates: list[Candidate],
        user_id: Any,
        history_len: int,
        cold_item_ids: set[Any],
        top_m: int,
    ) -> list[ScoredCandidate]:
        raise NotImplementedError
