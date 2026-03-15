from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from source.application.use_cases.training.stages.evaluate import evaluate_pipeline


class DummyLogger:
    def start_step(self, *args, **kwargs):
        return None

    def progress(self, *args, **kwargs):
        return None

    def end_step(self, *args, **kwargs):
        return None


class DummyPreRanker:
    def rank(self, candidates, user_id, history_len, cold_item_ids, top_m):
        from source.domain.entities import ScoredCandidate

        out = []
        for cand in candidates[:top_m]:
            out.append(
                ScoredCandidate(
                    user_id=user_id,
                    item_id=cand.item_id,
                    source=cand.source,
                    base_score=cand.score,
                    pre_score=float(cand.features.get("total_score", cand.score)),
                    features=dict(cand.features),
                )
            )
        return out


class DummyFinalRanker:
    def rank(self, candidates, user_id, top_k):
        from source.domain.entities import FinalItem

        return [
            FinalItem(
                user_id=user_id,
                item_id=item.item_id,
                source=item.source,
                final_score=item.pre_score,
                features=item.features,
            )
            for item in candidates[:top_k]
        ]


def test_evaluate_pipeline_reports_stage_metrics() -> None:
    data = {
        "books": pd.DataFrame({"item_id": [1, 2, 3, 4]}),
        "local_train": pd.DataFrame({"user_id": ["u1"], "item_id": [1]}),
        "local_val": pd.DataFrame({"user_id": ["u1"], "item_id": [2]}),
    }
    stage1 = {
        "pop_items": [3, 2],
        "pop_scores": {3: 1.0, 2: 0.5},
        "cf_neighbors": {1: [(2, 0.9)]},
        "content_similar": {1: [(2, 0.8), (4, 0.1)]},
        "item_metadata": {
            1: {"authors": ["A"], "series": [], "tags": ["fantasy"]},
            2: {"authors": ["A"], "series": [], "tags": ["fantasy"]},
            3: {"authors": ["B"], "series": [], "tags": ["drama"]},
        },
        "author_index": {"A": [1, 2], "B": [3]},
        "series_index": {},
        "tag_index": {"fantasy": [1, 2], "drama": [3]},
    }
    cmd = SimpleNamespace(
        eval_users_limit=10,
        candidate_pool_size=10,
        candidate_per_source_limit=5,
        pre_top_m=3,
        final_top_k=2,
    )

    metrics = evaluate_pipeline(
        data=data,
        stage1=stage1,
        stage2_model=DummyPreRanker(),
        stage3_model=DummyFinalRanker(),
        cmd=cmd,
        logger=DummyLogger(),
    )

    assert metrics["candidate_recall@10"] >= metrics["prerank_recall@3"] >= metrics["recall@2"]
    assert "cold_candidate_recall@10" in metrics
    assert "cold_prerank_recall@3" in metrics
