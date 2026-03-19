from __future__ import annotations

from source.domain.entities import Candidate, FinalItem
from source.infrastructure.processing.postprocessing.default_postprocessor import (
    DefaultPostprocessor,
)
from source.infrastructure.ranking.finalrank.policy_final_reranker import (
    PolicyFinalReranker,
    PolicyFinalRerankerConfig,
)
from source.infrastructure.ranking.prerank.catboost_preranker import (
    CatBoostPreRanker,
    CatBoostPreRankerConfig,
)
from source.infrastructure.ranking.prerank.feature_builder import FeatureBuilder
from source.infrastructure.ranking.prerank.linear_preranker import (
    LinearPreRanker,
    LinearPreRankerConfig,
)


class FakeProbModel:
    def predict_proba(self, rows):
        if hasattr(rows, "to_dict"):
            payload = rows.to_dict(orient="records")
        else:
            payload = rows
        out = []
        for row in payload:
            score = (
                0.1
                + 0.6 * float(row.get("score_cold", 0.0))
                + 0.3 * float(row.get("metadata_overlap", 0.0))
            )
            score = max(0.0, min(1.0, score))
            out.append([1.0 - score, score])
        return out


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
        Candidate(
            user_id="u1",
            item_id=1,
            source="cf|pop",
            score=0.2,
            features={
                "score_cf": 0.2,
                "rank_cf": 1.0,
                "score_pop": 0.2,
                "rank_pop": 2.0,
                "source_count": 2.0,
            },
        ),
        Candidate(
            user_id="u1",
            item_id=2,
            source="content",
            score=1.2,
            features={"score_content": 1.2, "rank_content": 1.0, "source_count": 1.0},
        ),
    ]
    rows = builder.build(
        candidates=candidates, user_id="u1", history_len=10, cold_item_ids={2}
    )

    by_id = {row.item_id: row for row in rows}
    assert by_id[1].features["score_cf"] == 1.0
    assert by_id[1].features["score_pop"] == 1.0
    assert by_id[1].features["score_content"] == 0.0
    assert by_id[2].features["is_cold_item"] == 1.0
    assert by_id[1].features["history_len_norm"] == 0.2
    assert by_id[1].features["score_norm"] == 0.0
    assert by_id[2].features["score_norm"] == 1.0
    assert by_id[2].features["total_score_norm"] == 1.0
    assert by_id[1].features["source_count_norm"] == 0.5


def test_linear_preranker_returns_top_m_sorted_by_pre_score() -> None:
    ranker = LinearPreRanker(
        cfg=LinearPreRankerConfig(
            w_base=1.0,
            w_total_score=0.0,
            w_cf=0.0,
            w_content=0.0,
            w_pop=0.0,
            w_cold_source=0.0,
            w_cold_flag=0.0,
            w_history=0.0,
            w_source_count=0.0,
            w_popularity=0.0,
            w_metadata_overlap=0.0,
            w_rank=0.0,
        )
    )
    candidates = [
        Candidate(
            user_id="u1",
            item_id=1,
            source="cf",
            score=0.2,
            features={"score_cf": 0.2, "rank_cf": 3.0},
        ),
        Candidate(
            user_id="u1",
            item_id=2,
            source="cf",
            score=0.9,
            features={"score_cf": 0.9, "rank_cf": 1.0},
        ),
        Candidate(
            user_id="u1",
            item_id=3,
            source="cf",
            score=0.4,
            features={"score_cf": 0.4, "rank_cf": 2.0},
        ),
    ]
    out = ranker.rank(
        candidates=candidates, user_id="u1", history_len=5, cold_item_ids=set(), top_m=2
    )
    assert [x.item_id for x in out] == [2, 3]


def test_linear_preranker_can_promote_cold_candidate_with_overlap_signal() -> None:
    ranker = LinearPreRanker(
        cfg=LinearPreRankerConfig(
            w_base=0.0,
            w_total_score=0.0,
            w_cf=0.0,
            w_content=0.0,
            w_pop=0.0,
            w_cold_source=0.5,
            w_cold_flag=0.2,
            w_history=0.0,
            w_source_count=0.0,
            w_popularity=0.0,
            w_metadata_overlap=0.5,
            w_rank=0.0,
        )
    )
    candidates = [
        Candidate(
            user_id="u1",
            item_id=10,
            source="pop",
            score=1.0,
            features={
                "score_pop": 1.0,
                "rank_pop": 1.0,
                "total_score": 1.0,
                "source_count": 1.0,
            },
        ),
        Candidate(
            user_id="u1",
            item_id=20,
            source="cold",
            score=0.5,
            features={
                "score_cold": 0.5,
                "rank_cold": 1.0,
                "metadata_overlap": 4.0,
                "total_score": 0.5,
                "source_count": 1.0,
            },
        ),
    ]

    out = ranker.rank(
        candidates=candidates, user_id="u1", history_len=5, cold_item_ids={20}, top_m=2
    )
    assert [x.item_id for x in out] == [20, 10]


def test_linear_preranker_returns_empty_for_non_positive_top_m() -> None:
    ranker = LinearPreRanker()
    out = ranker.rank(
        candidates=[], user_id="u1", history_len=0, cold_item_ids=set(), top_m=0
    )
    assert out == []


def test_catboost_preranker_ranks_candidates_from_model_scores() -> None:
    ranker = CatBoostPreRanker(model=FakeProbModel(), cfg=CatBoostPreRankerConfig())
    candidates = [
        Candidate(
            user_id="u1",
            item_id=10,
            source="pop",
            score=1.0,
            features={
                "score_pop": 1.0,
                "rank_pop": 1.0,
                "total_score": 1.0,
                "source_count": 1.0,
            },
        ),
        Candidate(
            user_id="u1",
            item_id=20,
            source="cold",
            score=0.5,
            features={
                "score_cold": 0.8,
                "rank_cold": 1.0,
                "metadata_overlap": 4.0,
                "total_score": 0.5,
                "source_count": 1.0,
            },
        ),
    ]

    out = ranker.rank(
        candidates=candidates, user_id="u1", history_len=5, cold_item_ids={20}, top_m=2
    )
    assert [x.item_id for x in out] == [20, 10]


def test_policy_final_reranker_balances_source_and_keeps_cold_item() -> None:
    ranker = PolicyFinalReranker(
        cfg=PolicyFinalRerankerConfig(
            source_bias={"pop": 0.0, "cold": 0.02},
            source_repeat_penalty=0.2,
            cold_item_boost=0.1,
            metadata_overlap_boost=0.05,
            popularity_penalty=0.05,
            target_cold_items=1,
        )
    )
    from source.domain.entities import ScoredCandidate

    candidates = [
        ScoredCandidate(
            user_id="u1",
            item_id=1,
            source="pop",
            base_score=1.0,
            pre_score=0.95,
            features={"item_popularity": 1.0},
        ),
        ScoredCandidate(
            user_id="u1",
            item_id=2,
            source="pop",
            base_score=0.9,
            pre_score=0.9,
            features={"item_popularity": 0.9},
        ),
        ScoredCandidate(
            user_id="u1",
            item_id=3,
            source="cold",
            base_score=0.7,
            pre_score=0.78,
            features={
                "is_cold_item": 1.0,
                "metadata_overlap": 0.8,
                "item_popularity": 0.1,
            },
        ),
    ]

    out = ranker.rank(candidates=candidates, user_id="u1", top_k=3)
    top2 = [item.item_id for item in out][:2]
    assert 3 in top2
    assert 1 in top2


def test_policy_final_reranker_injects_relevant_cold_item_into_tail() -> None:
    ranker = PolicyFinalReranker(
        cfg=PolicyFinalRerankerConfig(
            source_bias={"pop": 0.0, "cold": 0.02},
            source_repeat_penalty=0.0,
            cold_item_boost=0.0,
            metadata_overlap_boost=0.0,
            popularity_penalty=0.02,
            target_cold_items=1,
            max_injected_cold_items=1,
            cold_injection_min_metadata_overlap=2.0,
            cold_injection_max_score_gap=0.1,
        )
    )

    from source.domain.entities import ScoredCandidate

    candidates = [
        ScoredCandidate(
            user_id="u1",
            item_id=10,
            source="pop",
            base_score=1.0,
            pre_score=0.96,
            features={"item_popularity": 1.0},
        ),
        ScoredCandidate(
            user_id="u1",
            item_id=11,
            source="pop",
            base_score=0.94,
            pre_score=0.93,
            features={"item_popularity": 0.9},
        ),
        ScoredCandidate(
            user_id="u1",
            item_id=12,
            source="content",
            base_score=0.9,
            pre_score=0.91,
            features={"item_popularity": 0.5},
        ),
        ScoredCandidate(
            user_id="u1",
            item_id=20,
            source="cold",
            base_score=0.82,
            pre_score=0.84,
            features={
                "is_cold_item": 1.0,
                "metadata_overlap": 3.0,
                "item_popularity": 0.05,
                "score_cold": 0.8,
            },
        ),
    ]

    out = ranker.rank(candidates=candidates, user_id="u1", top_k=3)
    assert [item.item_id for item in out] == [10, 11, 20]


def test_policy_final_reranker_skips_weak_cold_injection() -> None:
    ranker = PolicyFinalReranker(
        cfg=PolicyFinalRerankerConfig(
            source_bias={"pop": 0.0, "cold": 0.02},
            source_repeat_penalty=0.0,
            cold_item_boost=0.02,
            metadata_overlap_boost=0.04,
            popularity_penalty=0.02,
            target_cold_items=1,
            max_injected_cold_items=1,
            cold_injection_min_metadata_overlap=2.0,
            cold_injection_max_score_gap=0.05,
        )
    )

    from source.domain.entities import ScoredCandidate

    candidates = [
        ScoredCandidate(
            user_id="u1",
            item_id=10,
            source="pop",
            base_score=1.0,
            pre_score=0.96,
            features={"item_popularity": 1.0},
        ),
        ScoredCandidate(
            user_id="u1",
            item_id=11,
            source="content",
            base_score=0.95,
            pre_score=0.93,
            features={"item_popularity": 0.8},
        ),
        ScoredCandidate(
            user_id="u1",
            item_id=20,
            source="cold",
            base_score=0.6,
            pre_score=0.7,
            features={
                "is_cold_item": 1.0,
                "metadata_overlap": 1.0,
                "item_popularity": 0.05,
                "score_cold": 0.8,
            },
        ),
    ]

    out = ranker.rank(candidates=candidates, user_id="u1", top_k=2)
    assert [item.item_id for item in out] == [10, 11]
