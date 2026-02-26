from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import pandas as pd

from models.content_tfidf import ContentTfidfRecommender
from models.hybrid import HybridRecommender
from models.item2item import Item2ItemRecommender
from models.top_popular import TopPopularRecommender
from src.data_bundle import DataBundle, load_data_bundle
from src.evaluator import evaluate_predictions


@dataclass
class RunnerConfig:
    data_root: str = "data/data06"
    split_name: str = "local_v1"
    sample_users_n: Optional[int] = None
    seed: int = 42
    k: int = 10
    content_candidate_top_n: int = 200
    item2item_candidate_top_n: int = 200
    hybrid_cf_weight: float = 0.5
    hybrid_content_weight: float = 0.35
    hybrid_pop_weight: float = 0.15
    hybrid_cf_top_n: int = 300
    hybrid_content_top_n: int = 300
    hybrid_pop_top_n: int = 300


def _evaluate(model_name: str, predictions: pd.DataFrame,
    bundle: DataBundle, k: int, extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:

    metrics = evaluate_predictions(
        eval_ground_truth=bundle.eval_ground_truth,
        predictions=predictions,
        catalog_size=bundle.catalog_size,
        warm_item_ids=bundle.warm_item_ids,
        cold_item_ids=bundle.cold_item_ids,
        k=k,
    )
    row: dict[str, Any] = {"model": model_name}
    row.update(metrics)
    if extra:
        row.update(extra)
    return row


def run_training_pipeline(config: RunnerConfig) -> tuple[pd.DataFrame, DataBundle, dict[str, Any]]:

    bundle = load_data_bundle(
        data_root=config.data_root,
        split_name=config.split_name,
        sample_users_n=config.sample_users_n,
        seed=config.seed,
    )

    rows: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}

    top_popular = TopPopularRecommender().fit(bundle.local_train)
    trained_models["top_popular"] = top_popular
    top_popular_preds = top_popular.recommend(
        user_ids=bundle.eval_users,
        seen_items_by_user=bundle.seen_items_by_user,
        k=config.k,
    )
    rows.append(_evaluate("top_popular", top_popular_preds, bundle, config.k))

    content = ContentTfidfRecommender().fit(
        item_text=bundle.item_text,
        item_popularity=bundle.item_popularity,
    )
    trained_models["content_tfidf"] = content
    content_preds = content.recommend(
        user_ids=bundle.eval_users,
        seen_items_by_user=bundle.seen_items_by_user,
        k=config.k,
        candidate_top_n=config.content_candidate_top_n,
    )
    rows.append(
        _evaluate(
            "content_tfidf",
            content_preds,
            bundle,
            config.k,
            extra={"candidate_top_n": config.content_candidate_top_n},
        )
    )

    item2item = Item2ItemRecommender().fit(
        local_train=bundle.local_train,
        item_popularity=bundle.item_popularity,
    )
    trained_models["item2item"] = item2item
    item2item_preds = item2item.recommend(
        user_ids=bundle.eval_users,
        seen_items_by_user=bundle.seen_items_by_user,
        k=config.k,
        candidate_top_n=config.item2item_candidate_top_n,
    )
    rows.append(
        _evaluate(
            "item2item",
            item2item_preds,
            bundle,
            config.k,
            extra={"candidate_top_n": config.item2item_candidate_top_n},
        )
    )

    hybrid = HybridRecommender(
        cf_weight=config.hybrid_cf_weight,
        content_weight=config.hybrid_content_weight,
        pop_weight=config.hybrid_pop_weight,
        cf_top_n=config.hybrid_cf_top_n,
        content_top_n=config.hybrid_content_top_n,
        pop_top_n=config.hybrid_pop_top_n,
    ).fit(
        local_train=bundle.local_train,
        item_text=bundle.item_text,
        item_popularity=bundle.item_popularity,
    )
    trained_models["hybrid"] = hybrid
    hybrid_preds = hybrid.recommend(
        user_ids=bundle.eval_users,
        seen_items_by_user=bundle.seen_items_by_user,
        k=config.k,
    )
    rows.append(
        _evaluate(
            "hybrid",
            hybrid_preds,
            bundle,
            config.k,
            extra={
                "cf_weight": config.hybrid_cf_weight,
                "content_weight": config.hybrid_content_weight,
                "pop_weight": config.hybrid_pop_weight,
                "cf_top_n": config.hybrid_cf_top_n,
                "content_top_n": config.hybrid_content_top_n,
                "pop_top_n": config.hybrid_pop_top_n,
            },
        )
    )

    metrics_df = pd.DataFrame(rows)
    metric_priority = [
        "model",
        f"ndcg@{config.k}",
        f"recall@{config.k}",
        f"coverage@{config.k}",
        f"warm_ndcg@{config.k}",
        f"warm_recall@{config.k}",
        f"cold_ndcg@{config.k}",
        f"cold_recall@{config.k}",
        "eval_users",
        "warm_eval_users",
        "cold_eval_users",
    ]
    ordered = [c for c in metric_priority if c in metrics_df.columns]
    remaining = [c for c in metrics_df.columns if c not in ordered]
    metrics_df = metrics_df[ordered + remaining]

    return metrics_df, bundle, trained_models


def runner_config_to_dict(config: RunnerConfig) -> dict[str, Any]:
    return asdict(config)
