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
        cold_item_ids: set[Any] | None = None,
        max_items_per_token: int = 800,
        novelty_boost: float = 2.0,
        rare_item_boost: float = 1.4,
    ) -> None:
        self._item_metadata = item_metadata
        self._author_index = author_index
        self._series_index = series_index
        self._tag_index = tag_index
        self._popularity_scores = popularity_scores or {}
        self._cold_item_ids = set(cold_item_ids or set())
        self._max_items_per_token = max(1, int(max_items_per_token))
        self._novelty_boost = max(0.0, float(novelty_boost))
        self._rare_item_boost = max(0.0, float(rare_item_boost))
        self._cold_catalog = self._build_cold_catalog()

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

        if self._cold_catalog:
            return self._generate_from_cold_catalog(
                user_id=user_id,
                seen_items=seen_items,
                limit=limit,
                author_tokens=author_tokens,
                series_tokens=series_tokens,
                tag_tokens=tag_tokens,
            )

        score_map: dict[Any, float] = defaultdict(float)
        overlap_map: dict[Any, float] = defaultdict(float)

        self._collect(score_map, overlap_map, author_tokens, self._author_index, seen_items, base_weight=4.4)
        self._collect(score_map, overlap_map, series_tokens, self._series_index, seen_items, base_weight=4.0)
        self._collect(score_map, overlap_map, tag_tokens, self._tag_index, seen_items, base_weight=1.5)

        ranked_rows: list[tuple[Any, float, float, float]] = []
        for item_id, raw_score in score_map.items():
            pop = self._popularity(item_id)
            novelty = 1.0 + self._novelty_boost * (1.0 - pop)
            if pop <= 0.08:
                novelty *= 1.0 + self._rare_item_boost * (0.08 - pop) / 0.08
            if self._cold_item_ids and item_id in self._cold_item_ids:
                novelty *= 1.6
            score = float(raw_score) * novelty
            ranked_rows.append((item_id, score, overlap_map.get(item_id, 0.0), pop))
        ranked_rows.sort(
            key=lambda row: (
                row[0] in self._cold_item_ids if self._cold_item_ids else False,
                row[1],
                row[2],
                -row[3],
            ),
            reverse=True,
        )

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

    def _generate_from_cold_catalog(
        self,
        *,
        user_id: Any,
        seen_items: set[Any],
        limit: int,
        author_tokens: Counter[str],
        series_tokens: Counter[str],
        tag_tokens: Counter[str],
    ) -> list[Candidate]:
        ranked_rows: list[tuple[Any, float, float, float]] = []

        for item_id, meta in self._cold_catalog:
            if item_id in seen_items:
                continue

            author_overlap = sum(author_tokens.get(token, 0) for token in meta.get("authors", []))
            series_overlap = sum(series_tokens.get(token, 0) for token in meta.get("series", []))
            tag_overlap = sum(tag_tokens.get(token, 0) for token in meta.get("tags", []))

            raw_score = (
                4.8 * min(3.0, float(author_overlap))
                + 4.2 * min(3.0, float(series_overlap))
                + 1.4 * min(6.0, float(tag_overlap))
            )
            overlap = float(author_overlap + series_overlap + tag_overlap)
            if raw_score <= 0.0:
                continue

            pop = self._popularity(item_id)
            novelty = 1.0 + self._novelty_boost * (1.0 - pop)
            if pop <= 0.08:
                novelty *= 1.0 + self._rare_item_boost * (0.08 - pop) / 0.08
            score = float(raw_score) * novelty
            ranked_rows.append((item_id, score, overlap, pop))

        ranked_rows.sort(key=lambda row: (row[1], row[2], -row[3]), reverse=True)
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
            token_items = [item_id for item_id in index.get(str(token), []) if item_id not in seen_items]
            token_items.sort(
                key=lambda item_id: (
                    item_id not in self._cold_item_ids if self._cold_item_ids else False,
                    self._popularity(item_id),
                )
            )
            for item_id in token_items[: self._max_items_per_token]:
                if item_id in seen_items:
                    continue
                if self._cold_item_ids and item_id not in self._cold_item_ids:
                    continue
                increment = base_weight * min(3.0, float(freq)) * (1.0 + 0.75 * (1.0 - self._popularity(item_id)))
                score_map[item_id] += increment
                overlap_map[item_id] += 1.0

    def _build_cold_catalog(self) -> list[tuple[Any, dict[str, list[str]]]]:
        if not self._cold_item_ids:
            return []
        out: list[tuple[Any, dict[str, list[str]]]] = []
        for item_id in self._cold_item_ids:
            meta = self._item_metadata.get(item_id)
            if meta is None:
                continue
            out.append((item_id, meta))
        return out

    def _popularity(self, item_id: Any) -> float:
        value = float(self._popularity_scores.get(item_id, 0.0))
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value
