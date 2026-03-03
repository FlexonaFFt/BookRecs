from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from source.domain.entities import Candidate


class CandidateSourcePort(ABC):


    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate(self, user_id: Any, seen_items: set[Any], limit: int) -> list[Candidate]:
        raise NotImplementedError
