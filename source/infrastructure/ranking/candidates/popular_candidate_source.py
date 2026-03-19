from __future__ import annotations

from typing import Any

from source.application.ports import CandidateSourcePort
from source.domain.entities import Candidate


# Генерирует кандидатов из глобального рейтинга популярности.
class PopularCandidateSource(CandidateSourcePort):

    def __init__(
        self, top_items: list[Any], item_score: dict[Any, float] | None = None
    ) -> None:
        self._top_items = top_items
        self._item_score = item_score or {}

    @property
    def name(self) -> str:
        return "pop"

    def generate(
        self, user_id: Any, seen_items: set[Any], limit: int
    ) -> list[Candidate]:
        out: list[Candidate] = []
        for rank, item_id in enumerate(self._top_items, start=1):
            if item_id in seen_items:
                continue
            score = float(self._item_score.get(item_id, 1.0 / rank))
            out.append(
                Candidate(
                    user_id=user_id,
                    item_id=item_id,
                    source=self.name,
                    score=score,
                    features={
                        "score_pop": score,
                        "rank_pop": float(rank),
                        "item_popularity": score,
                    },
                )
            )
            if len(out) >= limit:
                break
        return out
