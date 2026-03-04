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
            generated = source.generate(
                user_id=cmd.user_id,
                seen_items=cmd.seen_items,
                limit=cmd.per_source_limit,
            )
            self._merge_candidates(merged, generated, cmd.seen_items)

        ranked = self._to_ranked(cmd.user_id, merged, cmd.pool_size)
        if len(ranked) >= cmd.pool_size:
            return ranked

        need = cmd.pool_size - len(ranked)
        refill = self._fallback.generate(
            user_id=cmd.user_id,
            seen_items=cmd.seen_items,
            limit=max(need * 2, cmd.per_source_limit),
        )
        self._merge_candidates(merged, refill, cmd.seen_items)
        return self._to_ranked(cmd.user_id, merged, cmd.pool_size)

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
                }
            else:
                merged[cand.item_id]["score"] += float(cand.score)
                merged[cand.item_id]["sources"].add(cand.source)

    @staticmethod
    def _to_ranked(user_id: Any, merged: dict[Any, dict[str, Any]], limit: int) -> list[Candidate]:
        rows = []
        for item_id, payload in merged.items():
            src = "|".join(sorted(payload["sources"]))
            rows.append((item_id, float(payload["score"]), src))
        rows.sort(key=lambda x: x[1], reverse=True)

        out: list[Candidate] = []
        for item_id, score, src in rows[:limit]:
            out.append(Candidate(user_id=user_id, item_id=item_id, source=src, score=score))
        return out
