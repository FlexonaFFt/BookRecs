from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from source.domain.entities import FinalItem


class PostProcessorPort(ABC):
    """
    Контракт post-processing после final ranking.
    """

    @abstractmethod
    def apply(
        self,
        items: list[FinalItem],
        seen_items: set[Any],
        top_k: int,
    ) -> list[FinalItem]:
        raise NotImplementedError
