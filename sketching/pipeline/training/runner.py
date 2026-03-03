from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import time
from typing import Any, Optional

import pandas as pd

from models.content_tfidf import ContentTfidfRecommender
from models.hybrid import HybridRecommender
from models.item2item import Item2ItemRecommender
from models.top_popular import TopPopularRecommender
from src.data_bundle import DataBundle, load_data_bundle
from src.evaluator import evaluate_predictions


logger = logging.getLogger(__name__)


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
    total_started = time.time()
    logger.info("Loading data bundle: data_root=%s split_name=%s", config.data_root, config.split_name)
    bundle = load_data_bundle(
        data_root=config.data_root,
        split_name=config.split_name,
        sample_users_n=config.sample_users_n,
        seed=config.seed,
    )
    logger.info(
        "Bundle loaded: train_rows=%s eval_users=%s catalog=%s warm_items=%s cold_items=%s",
        len(bundle.local_train),
        len(bundle.eval_users),
        bundle.catalog_size,
        len(bundle.warm_item_ids),
        len(bundle.cold_item_ids),
    )

    rows: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}

    started = time.time()
    logger.info("Training TopPopular")
    top_popular = TopPopularRecommender().fit(bundle.local_train)
    trained_models["top_popular"] = top_popular
    logger.info("Generating TopPopular predictions")
    top_popular_preds = top_popular.recommend(
        user_ids=bundle.eval_users,
        seen_items_by_user=bundle.seen_items_by_user,
        k=config.k,
    )
    rows.append(_evaluate("top_popular", top_popular_preds, bundle, config.k))
    logger.info("TopPopular done in %.2fs", time.time() - started)

    started = time.time()
    logger.info("Training ContentTFIDF")
    content = ContentTfidfRecommender().fit(
        item_text=bundle.item_text,
        item_popularity=bundle.item_popularity,
    )
    trained_models["content_tfidf"] = content
    logger.info("Generating ContentTFIDF predictions")
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
    logger.info("ContentTFIDF done in %.2fs", time.time() - started)

    started = time.time()
    logger.info("Training Item2Item")
    item2item = Item2ItemRecommender().fit(
        local_train=bundle.local_train,
        item_popularity=bundle.item_popularity,
    )
    trained_models["item2item"] = item2item
    logger.info("Generating Item2Item predictions")
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
    logger.info("Item2Item done in %.2fs", time.time() - started)

    started = time.time()
    logger.info(
        "Training Hybrid (cf=%.3f content=%.3f pop=%.3f)",
        config.hybrid_cf_weight,
        config.hybrid_content_weight,
        config.hybrid_pop_weight,
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
    logger.info("Generating Hybrid predictions")
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
    logger.info("Hybrid done in %.2fs", time.time() - started)

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
    logger.info("Training pipeline finished in %.2fs", time.time() - total_started)

    return metrics_df, bundle, trained_models


def runner_config_to_dict(config: RunnerConfig) -> dict[str, Any]:
    return asdict(config)
