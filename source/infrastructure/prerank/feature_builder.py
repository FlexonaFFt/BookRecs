from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from source.domain.entities import Candidate


@dataclass(frozen=True)
class FeatureRow:
    user_id: Any
    item_id: Any
    source: str
    base_score: float
    features: dict[str, float]


class FeatureBuilder:


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

        rows: list[FeatureRow] = []
        for cand in candidates:
            src_parts = set(cand.source.split("|"))
            score_norm = (float(cand.score) - mn) / denom

            rows.append(
                FeatureRow(
                    user_id=user_id,
                    item_id=cand.item_id,
                    source=cand.source,
                    base_score=float(cand.score),
                    features={
                        "score_norm": float(score_norm),
                        "score_cf": 1.0 if "cf" in src_parts else 0.0,
                        "score_content": 1.0 if "content" in src_parts else 0.0,
                        "score_pop": 1.0 if "pop" in src_parts else 0.0,
                        "is_cold_item": 1.0 if cand.item_id in cold_item_ids else 0.0,
                        "history_len_norm": hist_feature,
                    },
                )
            )
        return rows
