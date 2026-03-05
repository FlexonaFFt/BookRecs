from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from source.application.use_cases.ranking import (
    FinalRankUseCase,
    GenerateCandidatesUseCase,
    PreRankCandidatesUseCase,
    RecoFlowCommand,
    RecoFlowUseCase,
)
from source.infrastructure.inference.history import UserHistoryProvider
from source.infrastructure.inference.loader import ModelBundle
from source.infrastructure.inference.logger import InferenceRequestLogger
from source.infrastructure.processing.postprocessing import DefaultPostprocessor
from source.infrastructure.ranking.candidates import (
    CfCandidateSource,
    ContentCandidateSource,
    PopularCandidateSource,
)


@dataclass(frozen=True)
class InferenceRequest:
    user_id: Any
    top_k: int
    candidate_pool_size: int
    candidate_per_source_limit: int
    pre_top_m: int
    seen_items: set[Any]
    use_history: bool = True


class InferenceService:
    def __init__(
        self,
        *,
        bundle: ModelBundle,
        history: UserHistoryProvider,
        request_logger: InferenceRequestLogger,
    ) -> None:
        self._bundle = bundle
        self._history = history
        self._request_logger = request_logger

        stage1 = bundle.stage1
        self._pop_scores: dict[Any, float] = stage1["pop_scores"]
        self._pop_items: list[Any] = stage1["pop_items"]
        self._cf_neighbors: dict[Any, list[tuple[Any, float]]] = stage1["cf_neighbors"]
        self._content_similar: dict[Any, list[tuple[Any, float]]] = stage1["content_similar"]
        self._item_id_type = self._detect_item_id_type()

        self._cold_item_ids = self._build_cold_item_ids()
        self._flow = RecoFlowUseCase(
            stage1=GenerateCandidatesUseCase(
                sources=[
                    CfCandidateSource(self._cf_neighbors),
                    ContentCandidateSource(self._content_similar, popularity_scores=self._pop_scores),
                    PopularCandidateSource(self._pop_items, self._pop_scores),
                ],
                fallback_source=PopularCandidateSource(self._pop_items, self._pop_scores),
            ),
            stage2=PreRankCandidatesUseCase(preranker=bundle.stage2),
            stage3=FinalRankUseCase(final_ranker=bundle.stage3, postprocessor=DefaultPostprocessor()),
        )

    @property
    def model_dir(self) -> str:
        return self._bundle.model_dir

    @property
    def metrics_snapshot(self) -> dict[str, Any]:
        return self._bundle.metrics_snapshot

    def recommend(self, req: InferenceRequest) -> dict[str, Any]:
        started = time.time()
        seen = set(self._normalize_item_id(x) for x in req.seen_items)
        seen.discard(None)
        if req.use_history:
            seen |= self._history.get_seen_items(req.user_id)
            seen = set(self._normalize_item_id(x) for x in seen if x is not None)

        result = self._flow.execute(
            RecoFlowCommand(
                user_id=req.user_id,
                seen_items=seen,
                history_len=len(seen),
                cold_item_ids=self._cold_item_ids,
                candidate_pool_size=req.candidate_pool_size,
                candidate_per_source_limit=req.candidate_per_source_limit,
                pre_top_m=req.pre_top_m,
                final_top_k=req.top_k,
            )
        )

        latency_ms = int((time.time() - started) * 1000)
        response = {
            "user_id": req.user_id,
            "history_len": len(seen),
            "model_dir": self.model_dir,
            "metrics_snapshot": self.metrics_snapshot,
            "items": [
                {
                    "item_id": x.item_id,
                    "score": x.final_score,
                    "source": x.source,
                }
                for x in result.final_items
            ],
        }
        try:
            self._request_logger.log(
                {
                    "user_id": req.user_id,
                    "endpoint": "/v1/recommendations",
                    "request": {
                        "top_k": req.top_k,
                        "candidate_pool_size": req.candidate_pool_size,
                        "candidate_per_source_limit": req.candidate_per_source_limit,
                        "pre_top_m": req.pre_top_m,
                        "seen_items": list(req.seen_items),
                        "use_history": req.use_history,
                    },
                    "response": {"items_count": len(response["items"])},
                    "model_dir": self.model_dir,
                    "latency_ms": latency_ms,
                }
            )
        except Exception:
            pass
        return response

    def similar_items(self, item_id: Any, limit: int = 10) -> dict[str, Any]:
        norm_item_id = self._normalize_item_id(item_id)
        if norm_item_id is None:
            raise ValueError("item_id is invalid for current model id type")
        content_rows = self._content_similar.get(norm_item_id, [])[: max(1, limit)]
        cf_rows = self._cf_neighbors.get(norm_item_id, [])[: max(1, limit)]
        return {
            "item_id": norm_item_id,
            "content": [{"item_id": i, "score": s} for i, s in content_rows],
            "cf": [{"item_id": i, "score": s} for i, s in cf_rows],
        }

    def register_interaction(self, user_id: Any, item_id: Any, event_type: str = "implicit") -> None:
        norm_item_id = self._normalize_item_id(item_id)
        if norm_item_id is None:
            raise ValueError("item_id is invalid for current model id type")
        self._history.add_interaction(
            user_id=user_id,
            item_id=norm_item_id,
            event_type=event_type,
        )

    def _build_cold_item_ids(self) -> set[Any]:
        warm = set(self._pop_scores.keys())
        all_items: set[Any] = set()
        all_items.update(self._pop_items)
        all_items.update(self._cf_neighbors.keys())
        all_items.update(self._content_similar.keys())
        for neighbors in self._cf_neighbors.values():
            for item_id, _ in neighbors:
                all_items.add(item_id)
        for neighbors in self._content_similar.values():
            for item_id, _ in neighbors:
                all_items.add(item_id)
        return all_items - warm

    def _detect_item_id_type(self):
        for pool in [self._pop_items, list(self._pop_scores.keys()), list(self._content_similar.keys())]:
            if pool:
                return type(pool[0])
        return str

    def _normalize_item_id(self, item_id: Any) -> Any:
        if item_id is None:
            return None
        target = self._item_id_type
        if isinstance(item_id, target):
            return item_id
        if target is int:
            try:
                return int(item_id)
            except (TypeError, ValueError):
                return None
        if target is str:
            return str(item_id)
        return item_id
