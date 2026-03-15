from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from source.application.ports import FinalRankerPort
from source.domain.entities import FinalItem, ScoredCandidate


@dataclass(frozen=True)
class PolicyFinalRerankerConfig:
    source_bias: dict[str, float] = field(default_factory=dict)
    source_repeat_penalty: float = 0.05
    cold_item_boost: float = 0.08
    metadata_overlap_boost: float = 0.04
    popularity_penalty: float = 0.03
    target_cold_items: int = 1


class PolicyFinalReranker(FinalRankerPort):
    """Greedy post-ranking policy layer over pre-ranked candidates."""

    def __init__(self, cfg: PolicyFinalRerankerConfig | None = None) -> None:
        self._cfg = cfg or PolicyFinalRerankerConfig()

    @property
    def cfg(self) -> PolicyFinalRerankerConfig:
        return self._cfg

    def rank(
        self,
        candidates: list[ScoredCandidate],
        user_id: Any,
        top_k: int,
    ) -> list[FinalItem]:
        if top_k <= 0:
            return []

        selected = self._rerank(candidates=candidates, top_k=top_k)
        return [
            FinalItem(
                user_id=user_id,
                item_id=item.item_id,
                source=item.source,
                final_score=float(self._score(item, source_counts={})),
                features=item.features,
            )
            for item in selected
        ]

    def _rerank(self, candidates: list[ScoredCandidate], top_k: int) -> list[ScoredCandidate]:
        pool = list(candidates)
        selected: list[ScoredCandidate] = []
        source_counts: dict[str, int] = {}
        cold_selected = 0

        while pool and len(selected) < top_k:
            remaining_slots = top_k - len(selected)
            need_cold = max(0, self._cfg.target_cold_items - cold_selected)
            restrict_to_cold = need_cold > 0 and remaining_slots <= need_cold and any(self._is_cold(x) for x in pool)

            best_idx = 0
            best_score = None
            for idx, item in enumerate(pool):
                if restrict_to_cold and not self._is_cold(item):
                    continue
                score = self._score(item, source_counts=source_counts)
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = idx

            picked = pool.pop(best_idx)
            selected.append(picked)
            source_counts[picked.source] = source_counts.get(picked.source, 0) + 1
            if self._is_cold(picked):
                cold_selected += 1
        return selected

    def _score(self, item: ScoredCandidate, source_counts: dict[str, int]) -> float:
        features = item.features or {}
        source_penalty = self._cfg.source_repeat_penalty * source_counts.get(item.source, 0)
        return float(
            item.pre_score
            + self._cfg.source_bias.get(item.source, 0.0)
            + self._cfg.cold_item_boost * features.get("is_cold_item", 0.0)
            + self._cfg.metadata_overlap_boost * features.get("metadata_overlap", 0.0)
            - self._cfg.popularity_penalty * features.get("item_popularity", 0.0)
            - source_penalty
        )

    @staticmethod
    def _is_cold(item: ScoredCandidate) -> bool:
        return bool((item.features or {}).get("is_cold_item", 0.0) > 0.0)
