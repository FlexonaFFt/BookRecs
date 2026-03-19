from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from source.application.ports import FinalRankerPort
from source.domain.entities import FinalItem, ScoredCandidate


@dataclass(frozen=True)
# Хранит веса линейного реранкера финального этапа.
class LinearFinalRerankerConfig:
    w_pre_score: float = 0.9
    w_cf: float = 0.08
    w_content: float = 0.08
    w_pop: float = 0.05
    w_cold: float = 0.04
    w_history: float = 0.03
    source_repeat_penalty: float = 0.0
    source_bias: dict[str, float] = field(default_factory=dict)


# Выполняет финальное ранжирование линейной моделью с bias по источникам.
class LinearFinalReranker(FinalRankerPort):
    def __init__(self, cfg: LinearFinalRerankerConfig | None = None) -> None:
        self._cfg = cfg or LinearFinalRerankerConfig()

    @property
    def cfg(self) -> LinearFinalRerankerConfig:
        return self._cfg

    def rank(
        self,
        candidates: list[ScoredCandidate],
        user_id: Any,
        top_k: int,
    ) -> list[FinalItem]:
        if top_k <= 0:
            return []

        ranked = self._rerank_with_source_diversity(candidates=candidates, top_k=top_k)
        return [
            FinalItem(
                user_id=user_id,
                item_id=item.item_id,
                source=item.source,
                final_score=float(self._score(item)),
                features=item.features,
            )
            for item in ranked
        ]

    def _rerank_with_source_diversity(
        self, candidates: list[ScoredCandidate], top_k: int
    ) -> list[ScoredCandidate]:
        if self._cfg.source_repeat_penalty <= 0:
            return sorted(candidates, key=self._score, reverse=True)[:top_k]

        pool = list(candidates)
        selected: list[ScoredCandidate] = []
        source_counts: dict[str, int] = {}
        while pool and len(selected) < top_k:
            best_idx = 0
            best_score = None
            for i, item in enumerate(pool):
                penalty = self._cfg.source_repeat_penalty * source_counts.get(
                    item.source, 0
                )
                score = self._score(item) - penalty
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = i
            picked = pool.pop(best_idx)
            selected.append(picked)
            source_counts[picked.source] = source_counts.get(picked.source, 0) + 1
        return selected

    def _score(self, item: ScoredCandidate) -> float:
        f = item.features or {}
        return float(
            self._cfg.w_pre_score * item.pre_score
            + self._cfg.w_cf * f.get("score_cf", 0.0)
            + self._cfg.w_content * f.get("score_content", 0.0)
            + self._cfg.w_pop * f.get("score_pop", 0.0)
            + self._cfg.w_cold * f.get("is_cold_item", 0.0)
            + self._cfg.w_history * f.get("history_len_norm", 0.0)
            + self._cfg.source_bias.get(item.source, 0.0)
        )
