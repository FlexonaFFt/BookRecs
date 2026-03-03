from __future__ import annotations

from typing import Any

from source.application.ports import CandidateSourcePort
from source.domain.entities import Candidate


class SourceContent(CandidateSourcePort):
    #Content-based candidate source.
    #similar_items: {item_id: [(candidate_item_id, content_score), ...]}


    def __init__(self, similar_items: dict[Any, list[tuple[Any, float]]]) -> None:
        self._similar_items = similar_items

    @property
    def name(self) -> str:
        return "content"

    def generate(self, user_id: Any, seen_items: set[Any], limit: int) -> list[Candidate]:

        score_map: dict[Any, float] = {}

        for item_id in seen_items:
            for candidate_id, score in self._similar_items.get(item_id, []):
                if candidate_id in seen_items:
                    continue
                score_map[candidate_id] = score_map.get(candidate_id, 0.0) + float(score)

        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [
            Candidate(user_id=user_id, item_id=item_id, source=self.name, score=float(score))
            for item_id, score in ranked
        ]
