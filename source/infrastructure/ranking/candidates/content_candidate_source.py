from __future__ import annotations

from typing import Any

from source.application.ports import CandidateSourcePort
from source.domain.entities import Candidate
# Генерирует кандидатов по соседям контентной похожести.
class ContentCandidateSource(CandidateSourcePort):


    def __init__(
        self,
        similar_items: dict[Any, list[tuple[Any, float]]],
        second_hop_decay: float = 0.35,
        second_hop_limit_factor: int = 3,
    ) -> None:
        self._similar_items = similar_items
        self._second_hop_decay = max(0.0, float(second_hop_decay))
        self._second_hop_limit_factor = max(1, int(second_hop_limit_factor))

    @property
    def name(self) -> str:
        return "content"

    def generate(self, user_id: Any, seen_items: set[Any], limit: int) -> list[Candidate]:

        score_map: dict[Any, float] = {}
        first_hop: dict[Any, float] = {}

        for item_id in seen_items:
            for candidate_id, score in self._similar_items.get(item_id, []):
                if candidate_id in seen_items:
                    continue
                val = score_map.get(candidate_id, 0.0) + float(score)
                score_map[candidate_id] = val
                first_hop[candidate_id] = first_hop.get(candidate_id, 0.0) + float(score)

        # Расширяет пул кандидатов вторым шагом по графу контентной похожести.
        if self._second_hop_decay > 0 and first_hop:
            first_hop_ranked = sorted(first_hop.items(), key=lambda x: x[1], reverse=True)[: limit * self._second_hop_limit_factor]
            for seed_item, seed_score in first_hop_ranked:
                for candidate_id, score in self._similar_items.get(seed_item, []):
                    if candidate_id in seen_items:
                        continue
                    score_map[candidate_id] = score_map.get(candidate_id, 0.0) + float(seed_score) * float(score) * self._second_hop_decay

        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [
            Candidate(user_id=user_id, item_id=item_id, source=self.name, score=float(score))
            for item_id, score in ranked
        ]
