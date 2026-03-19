from __future__ import annotations

from typing import Any

from source.application.ports import FinalRankerPort
from source.domain.entities import FinalItem, ScoredCandidate


# Формирует финальный рейтинг на основе скора из этапа 2.
class FinalRankerBaseline(FinalRankerPort):
    """
    Template final ranker.

    Current behavior:
    - uses pre_score as final_score;
    - keeps source/features trace;
    - returns top_k items.
    """

    def rank(
        self,
        candidates: list[ScoredCandidate],
        user_id: Any,
        top_k: int,
    ) -> list[FinalItem]:
        if top_k <= 0:
            return []

        ranked = sorted(candidates, key=lambda c: c.pre_score, reverse=True)[:top_k]
        return [
            FinalItem(
                user_id=user_id,
                item_id=c.item_id,
                source=c.source,
                final_score=float(c.pre_score),
                features=c.features,
            )
            for c in ranked
        ]
