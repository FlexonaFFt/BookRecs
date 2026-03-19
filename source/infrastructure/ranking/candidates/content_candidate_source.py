from __future__ import annotations

from typing import Any

from source.application.ports import CandidateSourcePort
from source.domain.entities import Candidate


# Генерирует кандидатов по соседям контентной похожести.
class ContentCandidateSource(CandidateSourcePort):

    def __init__(
        self,
        similar_items: dict[Any, list[tuple[Any, float]]],
        popularity_scores: dict[Any, float] | None = None,
        cold_item_ids: set[Any] | None = None,
        second_hop_decay: float = 0.5,
        second_hop_limit_factor: int = 4,
        novelty_boost: float = 0.9,
        cold_quota_ratio: float = 0.55,
        cold_popularity_threshold: float = 0.08,
    ) -> None:
        self._similar_items = similar_items
        self._popularity_scores = popularity_scores or {}
        self._cold_item_ids = set(cold_item_ids or set())
        self._second_hop_decay = max(0.0, float(second_hop_decay))
        self._second_hop_limit_factor = max(1, int(second_hop_limit_factor))
        self._novelty_boost = max(0.0, float(novelty_boost))
        self._cold_quota_ratio = min(1.0, max(0.0, float(cold_quota_ratio)))
        self._cold_popularity_threshold = min(
            1.0, max(0.0, float(cold_popularity_threshold))
        )

    @property
    def name(self) -> str:
        return "content"

    def generate(
        self, user_id: Any, seen_items: set[Any], limit: int
    ) -> list[Candidate]:

        score_map: dict[Any, float] = {}
        first_hop: dict[Any, float] = {}

        for item_id in seen_items:
            for candidate_id, score in self._similar_items.get(item_id, []):
                if candidate_id in seen_items:
                    continue
                val = score_map.get(candidate_id, 0.0) + float(score)
                score_map[candidate_id] = val
                first_hop[candidate_id] = first_hop.get(candidate_id, 0.0) + float(
                    score
                )

        # Расширяет пул кандидатов вторым шагом по графу контентной похожести.
        if self._second_hop_decay > 0 and first_hop:
            first_hop_ranked = sorted(
                first_hop.items(), key=lambda x: x[1], reverse=True
            )[: limit * self._second_hop_limit_factor]
            for seed_item, seed_score in first_hop_ranked:
                for candidate_id, score in self._similar_items.get(seed_item, []):
                    if candidate_id in seen_items:
                        continue
                    score_map[candidate_id] = (
                        score_map.get(candidate_id, 0.0)
                        + float(seed_score) * float(score) * self._second_hop_decay
                    )

        # Усиливает long-tail/новые объекты, которые чаще являются cold-кандидатами.
        adjusted_rows: list[tuple[Any, float, float, bool]] = []
        for item_id, score in score_map.items():
            pop_score = self._popularity(item_id)
            is_cold = (
                item_id in self._cold_item_ids
                if self._cold_item_ids
                else pop_score <= self._cold_popularity_threshold
            )
            cold_boost = 1.45 if is_cold else 1.0
            adjusted_score = (
                float(score)
                * (1.0 + self._novelty_boost * (1.0 - pop_score))
                * cold_boost
            )
            adjusted_rows.append((item_id, adjusted_score, pop_score, is_cold))

        adjusted_rows.sort(key=lambda x: (x[3], x[1], -x[2]), reverse=True)
        ranked = self._apply_cold_quota(adjusted_rows, limit=limit)
        out: list[Candidate] = []
        for rank, (item_id, score) in enumerate(ranked, start=1):
            out.append(
                Candidate(
                    user_id=user_id,
                    item_id=item_id,
                    source=self.name,
                    score=float(score),
                    features={
                        "score_content": float(score),
                        "rank_content": float(rank),
                        "item_popularity": float(self._popularity(item_id)),
                    },
                )
            )
        return out

    def _apply_cold_quota(
        self, rows: list[tuple[Any, float, float, bool]], limit: int
    ) -> list[tuple[Any, float]]:
        if limit <= 0:
            return []
        cold_limit = int(limit * self._cold_quota_ratio)
        if cold_limit <= 0:
            return [(item_id, score) for item_id, score, _, _ in rows[:limit]]

        selected: list[tuple[Any, float]] = []
        used: set[Any] = set()

        for item_id, score, pop_score, is_cold in rows:
            if len(selected) >= cold_limit:
                break
            if not is_cold:
                continue
            selected.append((item_id, score))
            used.add(item_id)

        for item_id, score, _, _ in rows:
            if len(selected) >= limit:
                break
            if item_id in used:
                continue
            selected.append((item_id, score))
            used.add(item_id)
        return selected[:limit]

    def _popularity(self, item_id: Any) -> float:
        value = float(self._popularity_scores.get(item_id, 0.0))
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value
