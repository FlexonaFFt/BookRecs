from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from source.application.ports import CandidateSourcePort
from source.domain.entities import Candidate


class ColdCandidateSource(CandidateSourcePort):
    """Retrieves cold and long-tail items via metadata overlap with user history."""

    def __init__(
        self,
        *,
        item_metadata: dict[Any, dict[str, list[str]]],
        author_index: dict[str, list[Any]],
        series_index: dict[str, list[Any]],
        tag_index: dict[str, list[Any]],
        popularity_scores: dict[Any, float] | None = None,
        max_items_per_token: int = 200,
        novelty_boost: float = 0.8,
    ) -> None:
        self._item_metadata = item_metadata
        self._author_index = author_index
        self._series_index = series_index
        self._tag_index = tag_index
        self._popularity_scores = popularity_scores or {}
        self._max_items_per_token = max(1, int(max_items_per_token))
        self._novelty_boost = max(0.0, float(novelty_boost))

    @property
    def name(self) -> str:
        return "cold"

    def generate(self, user_id: Any, seen_items: set[Any], limit: int) -> list[Candidate]:
        if limit <= 0 or not seen_items:
            return []

        author_tokens: Counter[str] = Counter()
        series_tokens: Counter[str] = Counter()
        tag_tokens: Counter[str] = Counter()
        for item_id in seen_items:
            meta = self._item_metadata.get(item_id)
            if meta is None:
                continue
            author_tokens.update(meta.get("authors", []))
            series_tokens.update(meta.get("series", []))
            tag_tokens.update(meta.get("tags", []))

        score_map: dict[Any, float] = defaultdict(float)
        overlap_map: dict[Any, float] = defaultdict(float)

        self._collect(score_map, overlap_map, author_tokens, self._author_index, seen_items, base_weight=3.0)
        self._collect(score_map, overlap_map, series_tokens, self._series_index, seen_items, base_weight=2.5)
        self._collect(score_map, overlap_map, tag_tokens, self._tag_index, seen_items, base_weight=0.9)

        ranked_rows: list[tuple[Any, float, float, float]] = []
        for item_id, raw_score in score_map.items():
            pop = self._popularity(item_id)
            novelty = 1.0 + self._novelty_boost * (1.0 - pop)
            score = float(raw_score) * novelty
            ranked_rows.append((item_id, score, overlap_map.get(item_id, 0.0), pop))
        ranked_rows.sort(key=lambda row: row[1], reverse=True)

        out: list[Candidate] = []
        for rank, (item_id, score, overlap, pop) in enumerate(ranked_rows[:limit], start=1):
            out.append(
                Candidate(
                    user_id=user_id,
                    item_id=item_id,
                    source=self.name,
                    score=float(score),
                    features={
                        "score_cold": float(score),
                        "rank_cold": float(rank),
                        "metadata_overlap": float(overlap),
                        "item_popularity": float(pop),
                    },
                )
            )
        return out

    def _collect(
        self,
        score_map: dict[Any, float],
        overlap_map: dict[Any, float],
        tokens: Counter[str],
        index: dict[str, list[Any]],
        seen_items: set[Any],
        *,
        base_weight: float,
    ) -> None:
        for token, freq in tokens.items():
            if not token:
                continue
            for item_id in index.get(str(token), [])[: self._max_items_per_token]:
                if item_id in seen_items:
                    continue
                increment = base_weight * min(3.0, float(freq))
                score_map[item_id] += increment
                overlap_map[item_id] += 1.0

    def _popularity(self, item_id: Any) -> float:
        value = float(self._popularity_scores.get(item_id, 0.0))
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value
