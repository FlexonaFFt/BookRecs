from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from source.application.ports import PreRankerPort
from source.domain.entities import Candidate, ScoredCandidate


@dataclass(frozen=True)
class PreRankCandidatesCommand:
    user_id: Any
    candidates: list[Candidate]
    history_len: int
    cold_item_ids: set[Any]
    top_m: int = 300


class PreRankCandidatesUseCase:
    """Stage 2: reduce candidate pool to top-M."""

    def __init__(self, preranker: PreRankerPort) -> None:
        self._preranker = preranker

    def execute(self, cmd: PreRankCandidatesCommand) -> list[ScoredCandidate]:
        if cmd.top_m <= 0:
            raise ValueError("top_m must be > 0")
        if not cmd.candidates:
            return []

        ranked = self._preranker.rank(
            candidates=cmd.candidates,
            user_id=cmd.user_id,
            history_len=cmd.history_len,
            cold_item_ids=cmd.cold_item_ids,
            top_m=cmd.top_m,
        )
        if ranked:
            return ranked

        # Fallback: preserve top-M by Stage-1 score.
        base_ranked = sorted(cmd.candidates, key=lambda c: c.score, reverse=True)[: cmd.top_m]
        return [
            ScoredCandidate(
                user_id=c.user_id,
                item_id=c.item_id,
                source=c.source,
                base_score=float(c.score),
                pre_score=float(c.score),
                features={"fallback": 1.0},
            )
            for c in base_ranked
        ]
