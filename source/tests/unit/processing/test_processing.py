from __future__ import annotations

from source.domain.entities import Candidate, FinalItem
from source.infrastructure.processing.postprocessing.default_postprocessor import DefaultPostprocessor
from source.infrastructure.ranking.prerank.feature_builder import FeatureBuilder
from source.infrastructure.ranking.prerank.linear_preranker import LinearPreRanker, LinearPreRankerConfig


def test_default_postprocessor_filters_seen_and_duplicates_and_limits_top_k() -> None:
    items = [
        FinalItem(user_id="u1", item_id=1, source="cf", final_score=0.9),
        FinalItem(user_id="u1", item_id=1, source="pop", final_score=0.8),
        FinalItem(user_id="u1", item_id=2, source="pop", final_score=0.7),
        FinalItem(user_id="u1", item_id=3, source="content", final_score=0.6),
    ]
    out = DefaultPostprocessor().apply(items=items, seen_items={3}, top_k=2)
    assert [x.item_id for x in out] == [1, 2]


def test_default_postprocessor_returns_empty_when_top_k_non_positive() -> None:
    items = [FinalItem(user_id="u1", item_id=1, source="cf", final_score=0.9)]
    out = DefaultPostprocessor().apply(items=items, seen_items=set(), top_k=0)
    assert out == []


def test_feature_builder_builds_expected_flags_and_normalized_scores() -> None:
    builder = FeatureBuilder()
    candidates = [
        Candidate(user_id="u1", item_id=1, source="cf|pop", score=0.2),
        Candidate(user_id="u1", item_id=2, source="content", score=1.2),
    ]
    rows = builder.build(candidates=candidates, user_id="u1", history_len=10, cold_item_ids={2})

    by_id = {row.item_id: row for row in rows}
    assert by_id[1].features["score_cf"] == 1.0
    assert by_id[1].features["score_pop"] == 1.0
    assert by_id[1].features["score_content"] == 0.0
    assert by_id[2].features["is_cold_item"] == 1.0
    assert by_id[1].features["history_len_norm"] == 0.2
    assert by_id[1].features["score_norm"] == 0.0
    assert by_id[2].features["score_norm"] == 1.0


def test_linear_preranker_returns_top_m_sorted_by_pre_score() -> None:
    ranker = LinearPreRanker(
        cfg=LinearPreRankerConfig(
            w_base=1.0,
            w_cf=0.0,
            w_content=0.0,
            w_pop=0.0,
            w_cold=0.0,
            w_history=0.0,
        )
    )
    candidates = [
        Candidate(user_id="u1", item_id=1, source="cf", score=0.2),
        Candidate(user_id="u1", item_id=2, source="cf", score=0.9),
        Candidate(user_id="u1", item_id=3, source="cf", score=0.4),
    ]
    out = ranker.rank(candidates=candidates, user_id="u1", history_len=5, cold_item_ids=set(), top_m=2)
    assert [x.item_id for x in out] == [2, 3]


def test_linear_preranker_returns_empty_for_non_positive_top_m() -> None:
    ranker = LinearPreRanker()
    out = ranker.rank(candidates=[], user_id="u1", history_len=0, cold_item_ids=set(), top_m=0)
    assert out == []
