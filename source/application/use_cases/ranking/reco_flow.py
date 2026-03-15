from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from source.application.use_cases.ranking.final_rank import FinalRankCommand, FinalRankUseCase
from source.application.use_cases.ranking.generate_candidates import (
    GenerateCandidatesCommand,
    GenerateCandidatesUseCase,
)
from source.application.use_cases.ranking.prerank_candidates import (
    PreRankCandidatesCommand,
    PreRankCandidatesUseCase,
)
from source.application.use_cases.ranking.source_limits import (
    source_limits_for_stage1,
    source_min_quota_for_stage1,
)
from source.domain.entities import Candidate, FinalItem, ScoredCandidate


@dataclass(frozen=True)
# Содержит входные данные для оркестрации рекомендательного потока.
class RecoFlowCommand:
    user_id: Any
    seen_items: set[Any]
    history_len: int
    cold_item_ids: set[Any]
    candidate_pool_size: int = 1000
    candidate_per_source_limit: int = 300
    pre_top_m: int = 300
    final_top_k: int = 10


@dataclass(frozen=True)
# Содержит результат выполнения рекомендательного потока.
class RecoFlowResult:
    candidates: list[Candidate]
    preranked: list[ScoredCandidate]
    final_items: list[FinalItem]
# Реализует сценарий оркестрации рекомендательного потока.
class RecoFlowUseCase:
    """Unified recommendation orchestrator: Stage1 -> Stage2 -> Stage3."""

    def __init__(
        self,
        stage1: GenerateCandidatesUseCase,
        stage2: PreRankCandidatesUseCase,
        stage3: FinalRankUseCase,
    ) -> None:
        self._stage1 = stage1
        self._stage2 = stage2
        self._stage3 = stage3

    def execute(self, cmd: RecoFlowCommand) -> RecoFlowResult:
        source_limits = source_limits_for_stage1(
            history_len=cmd.history_len,
            per_source_limit=cmd.candidate_per_source_limit,
        )
        candidates = self._stage1.execute(
            GenerateCandidatesCommand(
                user_id=cmd.user_id,
                seen_items=cmd.seen_items,
                pool_size=cmd.candidate_pool_size,
                per_source_limit=cmd.candidate_per_source_limit,
                source_limits=source_limits,
                source_min_quota=source_min_quota_for_stage1(
                    history_len=cmd.history_len,
                    pool_size=cmd.candidate_pool_size,
                ),
            )
        )

        preranked = self._stage2.execute(
            PreRankCandidatesCommand(
                user_id=cmd.user_id,
                candidates=candidates,
                history_len=cmd.history_len,
                cold_item_ids=cmd.cold_item_ids,
                top_m=cmd.pre_top_m,
            )
        )

        final_items = self._stage3.execute(
            FinalRankCommand(
                user_id=cmd.user_id,
                candidates=preranked,
                seen_items=cmd.seen_items,
                top_k=cmd.final_top_k,
            )
        )

        return RecoFlowResult(
            candidates=candidates,
            preranked=preranked,
            final_items=final_items,
        )
