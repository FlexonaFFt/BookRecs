from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from source.application.ports import PreRankerPort
from source.domain.entities import Candidate, ScoredCandidate
from source.infrastructure.ranking.prerank.feature_builder import FeatureBuilder


@dataclass(frozen=True)
# Хранит веса линейной модели предранжирования.
class LinearPreRankerConfig:
    w_base: float = 1.0
    w_cf: float = 0.2
    w_content: float = 0.18
    w_pop: float = 0.08
    w_cold_source: float = 0.28
    w_cold_flag: float = 0.12
    w_history: float = 0.02
    w_source_count: float = 0.12
    w_popularity: float = 0.04
    w_metadata_overlap: float = 0.16
    w_rank: float = 0.18
# Считает скор кандидатов линейной комбинацией ручных признаков.
class LinearPreRanker(PreRankerPort):
    """Simple linear pre-ranker used as Stage-2 baseline."""

    def __init__(
        self,
        cfg: LinearPreRankerConfig | None = None,
        feature_builder: FeatureBuilder | None = None,
    ) -> None:
        self._cfg = cfg or LinearPreRankerConfig()
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
                + self._cfg.w_cold_source * f["score_cold"]
                + self._cfg.w_rank
                * (
                    f["rank_inv_cf"]
                    + f["rank_inv_content"]
                    + f["rank_inv_pop"]
                    + f["rank_inv_cold"]
                )
                + self._cfg.w_cold_flag * f["is_cold_item"]
                + self._cfg.w_history * f["history_len_norm"]
                + self._cfg.w_source_count * f["source_count_norm"]
                + self._cfg.w_popularity * f["item_popularity"]
                + self._cfg.w_metadata_overlap * f["metadata_overlap"]
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
