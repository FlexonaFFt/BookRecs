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

        ranked = sorted(
            candidates,
            key=self._score,
            reverse=True,
        )[:top_k]
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
