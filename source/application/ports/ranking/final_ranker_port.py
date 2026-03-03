from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from source.domain.entities import FinalItem, ScoredCandidate


class FinalRankerPort(ABC):
    """
    Контракт Stage 3 final ranker.
    """

    @abstractmethod
    def rank(
        self,
        candidates: list[ScoredCandidate],
        user_id: Any,
        top_k: int,
    ) -> list[FinalItem]:
        raise NotImplementedError
