from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from source.application.ports import FinalRankerPort, PostProcessorPort
from source.domain.entities import FinalItem, ScoredCandidate


@dataclass(frozen=True)
# Содержит входные данные команды финального ранжирования.
class FinalRankCommand:
    user_id: Any
    candidates: list[ScoredCandidate]
    seen_items: set[Any]
    top_k: int = 10
# Реализует сценарий финального ранжирования.
class FinalRankUseCase:
    """
    Stage 3: final ranking + postprocessing.
    """

    def __init__(
        self,
        final_ranker: FinalRankerPort,
        postprocessor: PostProcessorPort,
    ) -> None:
        self._final_ranker = final_ranker
        self._postprocessor = postprocessor

    def execute(self, cmd: FinalRankCommand) -> list[FinalItem]:
        if cmd.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not cmd.candidates:
            return []

        ranked = self._final_ranker.rank(
            candidates=cmd.candidates,
            user_id=cmd.user_id,
            top_k=max(cmd.top_k * 3, cmd.top_k),
        )
        if not ranked:
            return []

        return self._postprocessor.apply(
            items=ranked,
            seen_items=cmd.seen_items,
            top_k=cmd.top_k,
        )
