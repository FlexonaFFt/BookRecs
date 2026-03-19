from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from source.domain.entities import Candidate

PRE_RANK_FEATURE_COLUMNS = [
    "score_norm",
    "total_score_norm",
    "score_cf",
    "score_content",
    "score_pop",
    "score_cold",
    "rank_inv_cf",
    "rank_inv_content",
    "rank_inv_pop",
    "rank_inv_cold",
    "is_from_cf",
    "is_from_content",
    "is_from_pop",
    "is_from_cold",
    "source_count_norm",
    "item_popularity",
    "metadata_overlap",
    "is_cold_item",
    "history_len_norm",
]


@dataclass(frozen=True)
# Хранит признаки для одной пары пользователь-элемент.
class FeatureRow:
    user_id: Any
    item_id: Any
    source: str
    base_score: float
    features: dict[str, float]


# Строит строки признаков для этапа предранжирования.
class FeatureBuilder:
    """Builds deterministic Stage-2 features from Stage-1 candidates."""

    def build(
        self,
        candidates: list[Candidate],
        user_id: Any,
        history_len: int,
        cold_item_ids: set[Any],
    ) -> list[FeatureRow]:
        if not candidates:
            return []

        base_scores = [float(c.score) for c in candidates]
        mx = max(base_scores)
        mn = min(base_scores)
        denom = (mx - mn) if mx != mn else 1.0
        hist_feature = min(float(max(0, history_len)) / 50.0, 1.0)
        score_keys = ["score_cf", "score_content", "score_pop", "score_cold"]
        rank_keys = ["rank_cf", "rank_content", "rank_pop", "rank_cold"]
        max_by_key = {
            key: max(float(c.features.get(key, 0.0)) for c in candidates)
            for key in score_keys
        }
        max_rank_by_key = {
            key: max(float(c.features.get(key, 0.0)) for c in candidates)
            for key in rank_keys
        }
        max_total_score = max(
            float((c.features or {}).get("total_score", c.score)) for c in candidates
        )

        rows: list[FeatureRow] = []
        for cand in candidates:
            src_parts = set(cand.source.split("|"))
            score_norm = (float(cand.score) - mn) / denom
            extra = cand.features or {}
            feature_map = {
                "score_norm": float(score_norm),
                "total_score_norm": self._normalized_total_score(
                    extra, float(cand.score), max_total_score
                ),
                "score_cf": self._normalized_score(extra, "score_cf", max_by_key),
                "score_content": self._normalized_score(
                    extra, "score_content", max_by_key
                ),
                "score_pop": self._normalized_score(extra, "score_pop", max_by_key),
                "score_cold": self._normalized_score(extra, "score_cold", max_by_key),
                "rank_inv_cf": self._inverse_rank(extra, "rank_cf", max_rank_by_key),
                "rank_inv_content": self._inverse_rank(
                    extra, "rank_content", max_rank_by_key
                ),
                "rank_inv_pop": self._inverse_rank(extra, "rank_pop", max_rank_by_key),
                "rank_inv_cold": self._inverse_rank(
                    extra, "rank_cold", max_rank_by_key
                ),
                "is_from_cf": 1.0 if "cf" in src_parts else 0.0,
                "is_from_content": 1.0 if "content" in src_parts else 0.0,
                "is_from_pop": 1.0 if "pop" in src_parts else 0.0,
                "is_from_cold": 1.0 if "cold" in src_parts else 0.0,
                "source_count_norm": min(
                    float(extra.get("source_count", 1.0)) / 4.0, 1.0
                ),
                "item_popularity": float(extra.get("item_popularity", 0.0)),
                "metadata_overlap": min(
                    float(extra.get("metadata_overlap", 0.0)) / 5.0, 1.0
                ),
                "is_cold_item": 1.0 if cand.item_id in cold_item_ids else 0.0,
                "history_len_norm": hist_feature,
            }
            for key in PRE_RANK_FEATURE_COLUMNS:
                feature_map.setdefault(key, 0.0)

            rows.append(
                FeatureRow(
                    user_id=user_id,
                    item_id=cand.item_id,
                    source=cand.source,
                    base_score=float(cand.score),
                    features=feature_map,
                )
            )
        return rows

    @staticmethod
    def _normalized_score(
        features: dict[str, float],
        key: str,
        max_by_key: dict[str, float],
    ) -> float:
        value = float(features.get(key, 0.0))
        denom = max(float(max_by_key.get(key, 0.0)), 1e-9)
        return min(1.0, value / denom) if denom > 0 else 0.0

    @staticmethod
    def _inverse_rank(
        features: dict[str, float],
        key: str,
        max_by_key: dict[str, float],
    ) -> float:
        rank = float(features.get(key, 0.0))
        if rank <= 0:
            return 0.0
        max_rank = max(float(max_by_key.get(key, rank)), rank)
        if max_rank <= 1:
            return 1.0
        return 1.0 - ((rank - 1.0) / (max_rank - 1.0))

    @staticmethod
    def _normalized_total_score(
        features: dict[str, float], base_score: float, max_total_score: float
    ) -> float:
        value = float(features.get("total_score", base_score))
        denom = max(float(max_total_score), 1e-9)
        return min(1.0, value / denom) if denom > 0 else 0.0
