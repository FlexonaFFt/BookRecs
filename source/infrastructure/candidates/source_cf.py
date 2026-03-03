from __future__ import annotations

from typing import Any

from source.application.ports import CandidateSourcePort
from source.domain.entities import Candidate


class SourceCf(CandidateSourcePort):
    #Item2Item candidate source.
    #neighbors: {item_id: [(neighbor_item_id, similarity_score), ...]}
    #Берет neighbors словарь и по seen-айтемам накапливает score соседей.

    def __init__(self, neighbors: dict[Any, list[tuple[Any, float]]]) -> None:
        self._neighbors = neighbors

    @property
    def name(self) -> str:
        return "cf"

    def generate(self, user_id: Any, seen_items: set[Any], limit: int) -> list[Candidate]:
        score_map: dict[Any, float] = {}

        for item_id in seen_items:
            for neighbor_id, score in self._neighbors.get(item_id, []):
                if neighbor_id in seen_items:
                    continue
                score_map[neighbor_id] = score_map.get(neighbor_id, 0.0) + float(score)

        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [
            Candidate(user_id=user_id, item_id=item_id, source=self.name, score=float(score))
            for item_id, score in ranked
        ]
