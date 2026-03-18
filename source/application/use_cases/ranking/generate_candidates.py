from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from source.application.ports import CandidateSourcePort
from source.domain.entities import Candidate


@dataclass(frozen=True)
# Содержит входные данные команды генерации кандидатов.
class GenerateCandidatesCommand:
    user_id: Any
    seen_items: set[Any]
    pool_size: int = 1000
    per_source_limit: int = 300
    source_limits: dict[str, int] | None = None
    source_min_quota: dict[str, int] | None = None
    cold_tail_injection_count: int = 2
    cold_tail_min_metadata_overlap: float = 1.5
    cold_tail_max_score_gap: float = 0.12
# Реализует сценарий генерации кандидатов.
class GenerateCandidatesUseCase:
    """Stage 1: generate and merge candidates from multiple sources."""

    def __init__(
        self,
        sources: list[CandidateSourcePort],
        fallback_source: CandidateSourcePort,
    ) -> None:
        if not sources:
            raise ValueError("At least one candidate source is required")
        self._sources = sources
        self._fallback = fallback_source

    def execute(self, cmd: GenerateCandidatesCommand) -> list[Candidate]:
        if cmd.pool_size <= 0:
            raise ValueError("pool_size must be > 0")
        if cmd.per_source_limit <= 0:
            raise ValueError("per_source_limit must be > 0")

        merged: dict[Any, dict[str, Any]] = {}

        for source in self._sources:
            source_limit = cmd.per_source_limit
            if cmd.source_limits is not None:
                source_limit = int(cmd.source_limits.get(source.name, cmd.per_source_limit))
            if source_limit <= 0:
                continue
            generated = source.generate(
                user_id=cmd.user_id,
                seen_items=cmd.seen_items,
                limit=source_limit,
            )
            self._merge_candidates(merged, generated, cmd.seen_items)

        ranked = self._to_ranked(
            cmd.user_id,
            merged,
            cmd.pool_size,
            source_min_quota=cmd.source_min_quota,
            cold_tail_injection_count=cmd.cold_tail_injection_count,
            cold_tail_min_metadata_overlap=cmd.cold_tail_min_metadata_overlap,
            cold_tail_max_score_gap=cmd.cold_tail_max_score_gap,
        )
        if len(ranked) >= cmd.pool_size:
            return ranked

        need = cmd.pool_size - len(ranked)
        fallback_limit = max(need * 2, cmd.per_source_limit)
        if cmd.source_limits is not None:
            fallback_limit = max(
                fallback_limit,
                int(cmd.source_limits.get(self._fallback.name, cmd.per_source_limit)),
            )
        refill = self._fallback.generate(
            user_id=cmd.user_id,
            seen_items=cmd.seen_items,
            limit=fallback_limit,
        )
        self._merge_candidates(merged, refill, cmd.seen_items)
        return self._to_ranked(
            cmd.user_id,
            merged,
            cmd.pool_size,
            source_min_quota=cmd.source_min_quota,
            cold_tail_injection_count=cmd.cold_tail_injection_count,
            cold_tail_min_metadata_overlap=cmd.cold_tail_min_metadata_overlap,
            cold_tail_max_score_gap=cmd.cold_tail_max_score_gap,
        )

    @staticmethod
    def _merge_candidates(
        merged: dict[Any, dict[str, Any]],
        candidates: list[Candidate],
        seen_items: set[Any],
    ) -> None:
        for cand in candidates:
            if cand.item_id in seen_items:
                continue
            if cand.item_id not in merged:
                merged[cand.item_id] = {
                    "score": float(cand.score),
                    "sources": {cand.source},
                    "features": dict(cand.features),
                }
            else:
                merged[cand.item_id]["score"] += float(cand.score)
                merged[cand.item_id]["sources"].add(cand.source)
                merged[cand.item_id]["features"] = GenerateCandidatesUseCase._merge_features(
                    merged[cand.item_id]["features"],
                    cand.features,
                )

    @staticmethod
    def _to_ranked(
        user_id: Any,
        merged: dict[Any, dict[str, Any]],
        limit: int,
        source_min_quota: dict[str, int] | None = None,
        cold_tail_injection_count: int = 0,
        cold_tail_min_metadata_overlap: float = 0.0,
        cold_tail_max_score_gap: float = 0.0,
    ) -> list[Candidate]:
        rows = []
        for item_id, payload in merged.items():
            src = "|".join(sorted(payload["sources"]))
            features = dict(payload.get("features", {}))
            features["source_count"] = float(len(payload["sources"]))
            features["total_score"] = float(payload["score"])
            rows.append((item_id, float(payload["score"]), src, features))
        rows.sort(key=lambda x: x[1], reverse=True)

        selected_rows = GenerateCandidatesUseCase._apply_source_min_quota(
            rows=rows,
            limit=limit,
            source_min_quota=source_min_quota or {},
        )
        selected_rows = GenerateCandidatesUseCase._inject_cold_tail(
            rows=rows,
            selected_rows=selected_rows,
            limit=limit,
            cold_tail_injection_count=cold_tail_injection_count,
            cold_tail_min_metadata_overlap=cold_tail_min_metadata_overlap,
            cold_tail_max_score_gap=cold_tail_max_score_gap,
        )
        return [
            Candidate(user_id=user_id, item_id=item_id, source=src, score=score, features=features)
            for item_id, score, src, features in selected_rows
        ]

    @staticmethod
    def _apply_source_min_quota(
        rows: list[tuple[Any, float, str, dict[str, float]]],
        limit: int,
        source_min_quota: dict[str, int],
    ) -> list[tuple[Any, float, str, dict[str, float]]]:
        if limit <= 0:
            return []
        if not source_min_quota:
            return rows[:limit]

        selected: list[tuple[Any, float, str, dict[str, float]]] = []
        used: set[Any] = set()

        for source_name, quota in source_min_quota.items():
            need = max(0, int(quota))
            if need <= 0:
                continue
            for row in rows:
                item_id, _, src, _ = row
                if item_id in used:
                    continue
                src_parts = set(src.split("|"))
                if source_name not in src_parts:
                    continue
                selected.append(row)
                used.add(item_id)
                if len(selected) >= limit:
                    return selected[:limit]
                need -= 1
                if need <= 0:
                    break

        for row in rows:
            item_id = row[0]
            if item_id in used:
                continue
            selected.append(row)
            used.add(item_id)
            if len(selected) >= limit:
                break
        return selected[:limit]

    @staticmethod
    def _merge_features(base: dict[str, float], extra: dict[str, float]) -> dict[str, float]:
        merged = dict(base)
        for key, value in extra.items():
            numeric = float(value)
            if key.startswith("score_"):
                merged[key] = max(merged.get(key, 0.0), numeric)
                continue
            if key.startswith("rank_"):
                current = merged.get(key)
                merged[key] = numeric if current is None else min(float(current), numeric)
                continue
            if key in {"item_popularity"}:
                merged[key] = max(merged.get(key, 0.0), numeric)
                continue
            merged[key] = merged.get(key, 0.0) + numeric
        return merged

    @staticmethod
    def _inject_cold_tail(
        rows: list[tuple[Any, float, str, dict[str, float]]],
        selected_rows: list[tuple[Any, float, str, dict[str, float]]],
        limit: int,
        cold_tail_injection_count: int,
        cold_tail_min_metadata_overlap: float,
        cold_tail_max_score_gap: float,
    ) -> list[tuple[Any, float, str, dict[str, float]]]:
        if limit <= 0 or cold_tail_injection_count <= 0 or not selected_rows:
            return selected_rows[:limit]

        result = list(selected_rows[:limit])
        selected_ids = {row[0] for row in result}
        current_cold = sum(1 for row in result if GenerateCandidatesUseCase._is_cold_row(row))
        need = max(0, int(cold_tail_injection_count) - current_cold)
        if need <= 0:
            return result

        cold_pool = [
            row for row in rows
            if row[0] not in selected_ids and GenerateCandidatesUseCase._eligible_cold_row(row, cold_tail_min_metadata_overlap)
        ]
        if not cold_pool:
            return result

        cold_pool.sort(key=GenerateCandidatesUseCase._cold_priority, reverse=True)
        for cold_row in cold_pool:
            if need <= 0:
                break
            replace_idx = GenerateCandidatesUseCase._find_tail_replacement_index(result)
            if replace_idx is None:
                break
            replace_row = result[replace_idx]
            if (float(replace_row[1]) - float(cold_row[1])) > float(cold_tail_max_score_gap):
                continue
            result[replace_idx] = cold_row
            need -= 1
        return result[:limit]

    @staticmethod
    def _is_cold_row(row: tuple[Any, float, str, dict[str, float]]) -> bool:
        src_parts = set(row[2].split("|"))
        features = row[3] or {}
        return "cold" in src_parts or float(features.get("is_cold_item", 0.0)) > 0.0

    @staticmethod
    def _eligible_cold_row(
        row: tuple[Any, float, str, dict[str, float]],
        min_metadata_overlap: float,
    ) -> bool:
        if not GenerateCandidatesUseCase._is_cold_row(row):
            return False
        features = row[3] or {}
        return float(features.get("metadata_overlap", 0.0)) >= float(min_metadata_overlap)

    @staticmethod
    def _cold_priority(row: tuple[Any, float, str, dict[str, float]]) -> float:
        _, score, _, features = row
        return float(score) + 0.02 * float(features.get("metadata_overlap", 0.0)) + 0.01 * float(features.get("score_cold", 0.0))

    @staticmethod
    def _find_tail_replacement_index(rows: list[tuple[Any, float, str, dict[str, float]]]) -> int | None:
        for idx in range(len(rows) - 1, -1, -1):
            if not GenerateCandidatesUseCase._is_cold_row(rows[idx]):
                return idx
        return None
