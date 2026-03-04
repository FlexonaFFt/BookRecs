from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from source.domain.entities import Candidate, FinalItem, ScoredCandidate


class CandidateSourcePort(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate(self, user_id: Any, seen_items: set[Any], limit: int) -> list[Candidate]:
        raise NotImplementedError


class PreRankerPort(ABC):
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


class FinalRankerPort(ABC):
    @abstractmethod
    def rank(
        self,
        candidates: list[ScoredCandidate],
        user_id: Any,
        top_k: int,
    ) -> list[FinalItem]:
        raise NotImplementedError


class PostProcessorPort(ABC):
    @abstractmethod
    def apply(
        self,
        items: list[FinalItem],
        seen_items: set[Any],
        top_k: int,
    ) -> list[FinalItem]:
        raise NotImplementedError
