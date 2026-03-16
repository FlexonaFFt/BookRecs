from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from source.application.use_cases.ranking.final_rank import FinalRankCommand, FinalRankUseCase
from source.application.use_cases.ranking.generate_candidates import (
    GenerateCandidatesCommand,
    GenerateCandidatesUseCase,
)
from source.application.use_cases.ranking.prerank_candidates import (
    PreRankCandidatesCommand,
    PreRankCandidatesUseCase,
)
from source.application.use_cases.ranking.reco_flow import RecoFlowCommand, RecoFlowUseCase
from source.application.use_cases.ranking.source_limits import (
    source_limits_for_stage1,
    source_min_quota_for_stage1,
)
from source.domain.entities import Candidate, FinalItem, ScoredCandidate
from source.infrastructure.ranking.candidates import ColdCandidateSource, ContentCandidateSource


@dataclass
class StubCandidateSource:
    name: str
    items: list[tuple[Any, float] | tuple[Any, float, dict[str, float]]]

    def __post_init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, user_id: Any, seen_items: set[Any], limit: int) -> list[Candidate]:
        self.calls.append({"user_id": user_id, "seen_items": set(seen_items), "limit": limit})
        out: list[Candidate] = []
        for raw in self.items[:limit]:
            if len(raw) == 2:
                item_id, score = raw
                features: dict[str, float] = {}
            else:
                item_id, score, features = raw
            out.append(Candidate(user_id=user_id, item_id=item_id, source=self.name, score=score, features=features))
        return out


@dataclass
class StubPreRanker:
    result: list[ScoredCandidate]

    def __post_init__(self) -> None:
        self.calls = 0

    def rank(
        self,
        candidates: list[Candidate],
        user_id: Any,
        history_len: int,
        cold_item_ids: set[Any],
        top_m: int,
    ) -> list[ScoredCandidate]:
        self.calls += 1
        return self.result


@dataclass
class StubFinalRanker:
    result: list[FinalItem]

    def __post_init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def rank(self, candidates: list[ScoredCandidate], user_id: Any, top_k: int) -> list[FinalItem]:
        self.calls.append({"candidates": candidates, "user_id": user_id, "top_k": top_k})
        return self.result


@dataclass
class StubPostProcessor:
    result: list[FinalItem]

    def __post_init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply(self, items: list[FinalItem], seen_items: set[Any], top_k: int) -> list[FinalItem]:
        self.calls.append({"items": items, "seen_items": set(seen_items), "top_k": top_k})
        return self.result


@dataclass
class StubStage1:
    candidates: list[Candidate]

    def __post_init__(self) -> None:
        self.last_cmd = None

    def execute(self, cmd):
        self.last_cmd = cmd
        return self.candidates


@dataclass
class StubStage2:
    items: list[ScoredCandidate]

    def __post_init__(self) -> None:
        self.last_cmd = None

    def execute(self, cmd):
        self.last_cmd = cmd
        return self.items


@dataclass
class StubStage3:
    items: list[FinalItem]

    def __post_init__(self) -> None:
        self.last_cmd = None

    def execute(self, cmd):
        self.last_cmd = cmd
        return self.items


def _candidates() -> list[Candidate]:
    return [
        Candidate(user_id="u1", item_id=1, source="cf", score=0.3),
        Candidate(user_id="u1", item_id=2, source="pop", score=0.9),
        Candidate(user_id="u1", item_id=3, source="content", score=0.5),
    ]


def _scored_candidates() -> list[ScoredCandidate]:
    return [
        ScoredCandidate(user_id="u1", item_id=1, source="cf", base_score=0.5, pre_score=0.7, features={}),
        ScoredCandidate(user_id="u1", item_id=2, source="pop", base_score=0.4, pre_score=0.6, features={}),
    ]


def test_generate_candidates_merges_scores_and_sources_and_filters_seen() -> None:
    cf = StubCandidateSource(name="cf", items=[(1, 0.7), (2, 0.3)])
    pop = StubCandidateSource(name="pop", items=[(1, 0.1), (3, 0.9)])
    uc = GenerateCandidatesUseCase(sources=[cf, pop], fallback_source=pop)

    result = uc.execute(
        GenerateCandidatesCommand(
            user_id="u1",
            seen_items={2},
            pool_size=2,
            per_source_limit=10,
        )
    )

    assert [x.item_id for x in result] == [3, 1]
    by_id = {x.item_id: x for x in result}
    assert by_id[1].score == pytest.approx(0.8)
    assert by_id[1].source == "cf|pop"
    assert by_id[3].source == "pop"
    assert by_id[1].features["source_count"] == pytest.approx(2.0)
    assert by_id[1].features["total_score"] == pytest.approx(0.8)


def test_generate_candidates_uses_fallback_when_pool_is_short() -> None:
    cf = StubCandidateSource(name="cf", items=[(10, 1.0)])
    pop = StubCandidateSource(name="pop", items=[(11, 0.9), (12, 0.8), (13, 0.7)])
    uc = GenerateCandidatesUseCase(sources=[cf], fallback_source=pop)

    result = uc.execute(
        GenerateCandidatesCommand(
            user_id="u1",
            seen_items=set(),
            pool_size=3,
            per_source_limit=1,
        )
    )

    assert [x.item_id for x in result] == [10, 11, 12]
    assert len(pop.calls) == 1
    assert pop.calls[0]["limit"] >= 2


def test_generate_candidates_respects_source_limits() -> None:
    cf = StubCandidateSource(name="cf", items=[(1, 1.0), (2, 0.9)])
    pop = StubCandidateSource(name="pop", items=[(3, 0.8), (4, 0.7)])
    uc = GenerateCandidatesUseCase(sources=[cf, pop], fallback_source=pop)

    _ = uc.execute(
        GenerateCandidatesCommand(
            user_id="u1",
            seen_items=set(),
            pool_size=4,
            per_source_limit=5,
            source_limits={"cf": 1, "pop": 2},
        )
    )

    assert cf.calls[0]["limit"] == 1
    assert pop.calls[0]["limit"] == 2


def test_generate_candidates_merges_source_specific_features() -> None:
    cf = StubCandidateSource(name="cf", items=[])
    pop = StubCandidateSource(name="pop", items=[])
    cf.items = [(1, 0.9)]
    pop.items = [(1, 0.5)]

    def _cf_generate(user_id: Any, seen_items: set[Any], limit: int) -> list[Candidate]:
        return [
            Candidate(
                user_id=user_id,
                item_id=1,
                source="cf",
                score=0.9,
                features={"score_cf": 0.9, "rank_cf": 1.0},
            )
        ]

    def _pop_generate(user_id: Any, seen_items: set[Any], limit: int) -> list[Candidate]:
        return [
            Candidate(
                user_id=user_id,
                item_id=1,
                source="pop",
                score=0.5,
                features={"score_pop": 0.5, "rank_pop": 3.0, "item_popularity": 0.5},
            )
        ]

    cf.generate = _cf_generate  # type: ignore[method-assign]
    pop.generate = _pop_generate  # type: ignore[method-assign]
    uc = GenerateCandidatesUseCase(sources=[cf, pop], fallback_source=pop)

    result = uc.execute(
        GenerateCandidatesCommand(user_id="u1", seen_items=set(), pool_size=1, per_source_limit=5)
    )

    assert len(result) == 1
    features = result[0].features
    assert features["score_cf"] == pytest.approx(0.9)
    assert features["score_pop"] == pytest.approx(0.5)
    assert features["rank_cf"] == pytest.approx(1.0)
    assert features["rank_pop"] == pytest.approx(3.0)
    assert features["item_popularity"] == pytest.approx(0.5)
    assert features["total_score"] == pytest.approx(1.4)


def test_generate_candidates_enforces_min_quota_for_cold_source() -> None:
    pop = StubCandidateSource(name="pop", items=[(1, 1.0), (2, 0.95), (3, 0.9)])
    cold = StubCandidateSource(name="cold", items=[(10, 0.2), (11, 0.19)])
    uc = GenerateCandidatesUseCase(sources=[pop, cold], fallback_source=pop)

    result = uc.execute(
        GenerateCandidatesCommand(
            user_id="u1",
            seen_items=set(),
            pool_size=3,
            per_source_limit=5,
            source_min_quota={"cold": 1},
        )
    )

    assert len(result) == 3
    assert any("cold" in item.source.split("|") for item in result)


def test_generate_candidates_injects_relevant_cold_candidate_into_tail() -> None:
    pop = StubCandidateSource(name="pop", items=[(1, 1.0), (2, 0.95), (3, 0.9)])
    cold = StubCandidateSource(
        name="cold",
        items=[
            (10, 0.82, {"metadata_overlap": 3.0, "score_cold": 0.8, "is_cold_item": 1.0}),
            (11, 0.4, {"metadata_overlap": 0.5, "score_cold": 0.3, "is_cold_item": 1.0}),
        ],
    )
    uc = GenerateCandidatesUseCase(sources=[pop, cold], fallback_source=pop)

    result = uc.execute(
        GenerateCandidatesCommand(
            user_id="u1",
            seen_items=set(),
            pool_size=3,
            per_source_limit=5,
            source_min_quota={},
            cold_tail_injection_count=1,
            cold_tail_min_metadata_overlap=2.0,
            cold_tail_max_score_gap=0.1,
        )
    )

    assert [item.item_id for item in result] == [1, 2, 10]


def test_source_limits_for_stage1_favor_content_and_cold_for_short_history() -> None:
    short = source_limits_for_stage1(history_len=1, per_source_limit=100)
    medium = source_limits_for_stage1(history_len=4, per_source_limit=100)
    long = source_limits_for_stage1(history_len=10, per_source_limit=100)

    assert short == {"cf": 15, "content": 280, "cold": 260, "pop": 60}
    assert medium == {"cf": 45, "content": 210, "cold": 190, "pop": 70}
    assert long == {"cf": 85, "content": 160, "cold": 150, "pop": 60}


def test_source_min_quota_for_stage1_reserves_cold_and_content() -> None:
    short = source_min_quota_for_stage1(history_len=1, pool_size=1000)
    medium = source_min_quota_for_stage1(history_len=4, pool_size=1000)
    long = source_min_quota_for_stage1(history_len=10, pool_size=1000)

    assert short == {"content": 420, "cold": 280}
    assert medium == {"content": 320, "cold": 220}
    assert long == {"content": 240, "cold": 180}


def test_cold_candidate_source_uses_metadata_overlap_and_novelty() -> None:
    source = ColdCandidateSource(
        item_metadata={
            1: {"authors": ["A"], "series": ["S"], "tags": ["fantasy"]},
            2: {"authors": ["A"], "series": [], "tags": ["fantasy"]},
            3: {"authors": [], "series": ["S"], "tags": ["magic"]},
        },
        author_index={"A": [1, 2]},
        series_index={"S": [1, 3]},
        tag_index={"fantasy": [1, 2], "magic": [3]},
        popularity_scores={2: 0.1, 3: 0.0},
    )

    out = source.generate(user_id="u1", seen_items={1}, limit=5)

    assert [x.item_id for x in out] == [3, 2]
    by_id = {x.item_id: x for x in out}
    assert by_id[2].features["metadata_overlap"] > 0.0
    assert by_id[3].features["rank_cold"] == pytest.approx(1.0)
    assert by_id[3].features["score_cold"] > by_id[2].features["score_cold"]


def test_cold_candidate_source_prioritizes_low_popularity_items_with_same_token() -> None:
    source = ColdCandidateSource(
        item_metadata={
            1: {"authors": ["A"], "series": [], "tags": []},
            2: {"authors": ["A"], "series": [], "tags": []},
            3: {"authors": ["A"], "series": [], "tags": []},
        },
        author_index={"A": [1, 2, 3]},
        series_index={},
        tag_index={},
        popularity_scores={2: 0.7, 3: 0.01},
    )

    out = source.generate(user_id="u1", seen_items={1}, limit=2)

    assert [x.item_id for x in out][:2] == [3, 2]


def test_cold_candidate_source_scores_explicit_cold_catalog() -> None:
    source = ColdCandidateSource(
        item_metadata={
            1: {"authors": ["A"], "series": ["S"], "tags": ["fantasy"]},
            10: {"authors": ["A"], "series": [], "tags": []},
            11: {"authors": [], "series": ["S"], "tags": ["fantasy"]},
            12: {"authors": ["B"], "series": [], "tags": []},
        },
        author_index={"A": [1, 10]},
        series_index={"S": [1, 11]},
        tag_index={"fantasy": [1, 11]},
        popularity_scores={10: 0.05, 11: 0.01, 12: 0.0},
        cold_item_ids={10, 11, 12},
    )

    out = source.generate(user_id="u1", seen_items={1}, limit=3)

    assert [x.item_id for x in out][:2] == [11, 10]
    assert all(item.item_id in {10, 11, 12} for item in out)


def test_content_candidate_source_reserves_low_popularity_candidates() -> None:
    source = ContentCandidateSource(
        similar_items={
            1: [(2, 0.9), (3, 0.8), (4, 0.7)],
        },
        popularity_scores={2: 0.6, 3: 0.04, 4: 0.01},
    )

    out = source.generate(user_id="u1", seen_items={1}, limit=2)

    assert len(out) == 2
    assert any(item.item_id in {3, 4} for item in out)


def test_generate_candidates_validates_input() -> None:
    pop = StubCandidateSource(name="pop", items=[(1, 1.0)])
    uc = GenerateCandidatesUseCase(sources=[pop], fallback_source=pop)

    with pytest.raises(ValueError):
        uc.execute(GenerateCandidatesCommand(user_id="u", seen_items=set(), pool_size=0, per_source_limit=1))
    with pytest.raises(ValueError):
        uc.execute(GenerateCandidatesCommand(user_id="u", seen_items=set(), pool_size=1, per_source_limit=0))


def test_generate_candidates_requires_at_least_one_source() -> None:
    pop = StubCandidateSource(name="pop", items=[])
    with pytest.raises(ValueError):
        _ = GenerateCandidatesUseCase(sources=[], fallback_source=pop)


def test_prerank_returns_model_result_when_non_empty() -> None:
    model_rows = [
        ScoredCandidate(
            user_id="u1",
            item_id=99,
            source="cf",
            base_score=0.2,
            pre_score=0.8,
            features={"score_cf": 1.0},
        )
    ]
    uc = PreRankCandidatesUseCase(preranker=StubPreRanker(result=model_rows))
    out = uc.execute(
        PreRankCandidatesCommand(
            user_id="u1",
            candidates=_candidates(),
            history_len=2,
            cold_item_ids=set(),
            top_m=3,
        )
    )
    assert out == model_rows


def test_prerank_fallbacks_to_base_scores_when_model_returns_empty() -> None:
    uc = PreRankCandidatesUseCase(preranker=StubPreRanker(result=[]))
    out = uc.execute(
        PreRankCandidatesCommand(
            user_id="u1",
            candidates=_candidates(),
            history_len=2,
            cold_item_ids=set(),
            top_m=2,
        )
    )
    assert [x.item_id for x in out] == [2, 3]
    assert all(x.features.get("fallback") == 1.0 for x in out)


def test_prerank_returns_empty_when_no_candidates() -> None:
    uc = PreRankCandidatesUseCase(preranker=StubPreRanker(result=[]))
    out = uc.execute(
        PreRankCandidatesCommand(
            user_id="u1",
            candidates=[],
            history_len=0,
            cold_item_ids=set(),
            top_m=10,
        )
    )
    assert out == []


def test_prerank_validates_top_m() -> None:
    uc = PreRankCandidatesUseCase(preranker=StubPreRanker(result=[]))
    with pytest.raises(ValueError):
        uc.execute(
            PreRankCandidatesCommand(
                user_id="u1",
                candidates=_candidates(),
                history_len=0,
                cold_item_ids=set(),
                top_m=0,
            )
        )


def test_final_rank_executes_ranker_and_postprocessor() -> None:
    ranked = [
        FinalItem(user_id="u1", item_id=2, source="pop", final_score=0.95),
        FinalItem(user_id="u1", item_id=1, source="cf", final_score=0.90),
    ]
    out_items = [FinalItem(user_id="u1", item_id=2, source="pop", final_score=0.95)]
    ranker = StubFinalRanker(result=ranked)
    post = StubPostProcessor(result=out_items)

    uc = FinalRankUseCase(final_ranker=ranker, postprocessor=post)
    out = uc.execute(
        FinalRankCommand(
            user_id="u1",
            candidates=_scored_candidates(),
            seen_items={1},
            top_k=2,
        )
    )

    assert out == out_items
    assert ranker.calls[0]["top_k"] == 6
    assert post.calls[0]["seen_items"] == {1}
    assert post.calls[0]["top_k"] == 2


def test_final_rank_returns_empty_when_no_candidates() -> None:
    ranker = StubFinalRanker(result=[])
    post = StubPostProcessor(result=[])
    uc = FinalRankUseCase(final_ranker=ranker, postprocessor=post)
    out = uc.execute(FinalRankCommand(user_id="u1", candidates=[], seen_items=set(), top_k=10))
    assert out == []
    assert ranker.calls == []
    assert post.calls == []


def test_final_rank_validates_top_k() -> None:
    ranker = StubFinalRanker(result=[])
    post = StubPostProcessor(result=[])
    uc = FinalRankUseCase(final_ranker=ranker, postprocessor=post)
    with pytest.raises(ValueError):
        uc.execute(FinalRankCommand(user_id="u1", candidates=_scored_candidates(), seen_items=set(), top_k=0))


def test_reco_flow_executes_all_stages_and_passes_params() -> None:
    cands = [Candidate(user_id="u1", item_id=1, source="cf", score=1.0)]
    preranked = [ScoredCandidate(user_id="u1", item_id=1, source="cf", base_score=1.0, pre_score=1.0, features={})]
    final = [FinalItem(user_id="u1", item_id=1, source="cf", final_score=1.0)]

    stage1 = StubStage1(candidates=cands)
    stage2 = StubStage2(items=preranked)
    stage3 = StubStage3(items=final)
    flow = RecoFlowUseCase(stage1=stage1, stage2=stage2, stage3=stage3)

    cmd = RecoFlowCommand(
        user_id="u1",
        seen_items={10, 20},
        history_len=7,
        cold_item_ids={99},
        candidate_pool_size=123,
        candidate_per_source_limit=50,
        pre_top_m=33,
        final_top_k=5,
    )
    out = flow.execute(cmd)

    assert out.candidates == cands
    assert out.preranked == preranked
    assert out.final_items == final
    assert stage1.last_cmd.pool_size == 123
    assert stage2.last_cmd.top_m == 33
    assert stage3.last_cmd.top_k == 5
    assert stage1.last_cmd.source_limits["cf"] == int(50 * 0.85)
    assert stage1.last_cmd.source_limits["content"] == int(50 * 1.6)
    assert stage1.last_cmd.source_limits["cold"] == int(50 * 1.5)
    assert stage1.last_cmd.source_limits["pop"] == int(50 * 0.6)
    assert stage1.last_cmd.source_min_quota == {"content": 29, "cold": 22}


def test_reco_flow_uses_content_heavy_limits_for_short_history() -> None:
    stage1 = StubStage1(candidates=[])
    stage2 = StubStage2(items=[])
    stage3 = StubStage3(items=[])
    flow = RecoFlowUseCase(stage1=stage1, stage2=stage2, stage3=stage3)

    _ = flow.execute(
        RecoFlowCommand(
            user_id="u1",
            seen_items=set(),
            history_len=1,
            cold_item_ids=set(),
            candidate_pool_size=10,
            candidate_per_source_limit=100,
            pre_top_m=5,
            final_top_k=3,
        )
    )

    limits = stage1.last_cmd.source_limits
    assert limits["cf"] == 15
    assert limits["content"] == 280
    assert limits["cold"] == 260
    assert limits["pop"] == 60
    assert stage1.last_cmd.source_min_quota == {"content": 4, "cold": 2}
