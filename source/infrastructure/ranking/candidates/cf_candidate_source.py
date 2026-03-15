from __future__ import annotations

from typing import Any

from source.application.ports import CandidateSourcePort
from source.domain.entities import Candidate
# Генерирует кандидатов по соседям коллаборативной фильтрации.
class CfCandidateSource(CandidateSourcePort):

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
        out: list[Candidate] = []
        for rank, (item_id, score) in enumerate(ranked, start=1):
            out.append(
                Candidate(
                    user_id=user_id,
                    item_id=item_id,
                    source=self.name,
                    score=float(score),
                    features={
                        "score_cf": float(score),
                        "rank_cf": float(rank),
                    },
                )
            )
        return out
