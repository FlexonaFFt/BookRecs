from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from source.application.ports import PreRankerPort
from source.domain.entities import Candidate, ScoredCandidate
from source.infrastructure.prerank.feature_builder import FeatureBuilder


@dataclass(frozen=True)
class PreRankLinearConfig:
    w_base: float = 1.0
    w_cf: float = 0.25
    w_content: float = 0.2
    w_pop: float = 0.1
    w_cold: float = 0.15
    w_history: float = 0.02


class PreRankLinear(PreRankerPort):


    def __init__(
        self,
        cfg: PreRankLinearConfig | None = None,
        feature_builder: FeatureBuilder | None = None,
    ) -> None:
        self._cfg = cfg or PreRankLinearConfig()
        self._feature_builder = feature_builder or FeatureBuilder()

    def rank(
        self,
        candidates: list[Candidate],
        user_id: Any,
        history_len: int,
        cold_item_ids: set[Any],
        top_m: int,
    ) -> list[ScoredCandidate]:
        if top_m <= 0:
            return []
        rows = self._feature_builder.build(
            candidates=candidates,
            user_id=user_id,
            history_len=history_len,
            cold_item_ids=cold_item_ids,
        )
        if not rows:
            return []

        scored: list[ScoredCandidate] = []
        for row in rows:
            f = row.features
            pre_score = (
                self._cfg.w_base * f["score_norm"]
                + self._cfg.w_cf * f["score_cf"]
                + self._cfg.w_content * f["score_content"]
                + self._cfg.w_pop * f["score_pop"]
                + self._cfg.w_cold * f["is_cold_item"]
                + self._cfg.w_history * f["history_len_norm"]
            )

            scored.append(
                ScoredCandidate(
                    user_id=row.user_id,
                    item_id=row.item_id,
                    source=row.source,
                    base_score=row.base_score,
                    pre_score=float(pre_score),
                    features=f,
                )
            )

        scored.sort(key=lambda x: x.pre_score, reverse=True)
        return scored[:top_m]
