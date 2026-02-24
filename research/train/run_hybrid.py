import argparse
import json
import logging
from pathlib import Path
import time

try:
    from evaluator import evaluate_predictions
    from hybrid import HybridContentPopularRecommender
    from preprocessor import load_research_bundle_cache
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from research.train.evaluator import evaluate_predictions  # type: ignore
    from research.train.hybrid import HybridContentPopularRecommender  # type: ignore
    from research.train.preprocessor import load_research_bundle_cache  # type: ignore


logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# Запустить гибридный метод и сохранить метрики
def run_hybrid_experiment(
    *,
    cache_dir: str = "research/results/cache/local_v1",
    results_dir: str = "research/results",
    result_name: str = "hybrid_content_popular",
    k: int = 10,
    content_weight: float = 0.8,
    popularity_weight: float = 0.2,
    content_top_n: int = 200,
    pop_candidates_n: int = 200,
    max_features: int = 50000,
    min_df: int = 2,
) -> dict:
    started = time.time()
    bundle = load_research_bundle_cache(cache_dir)

    model = HybridContentPopularRecommender(
        content_weight=content_weight,
        popularity_weight=popularity_weight,
        content_top_n=content_top_n,
        pop_candidates_n=pop_candidates_n,
        max_features=max_features,
        min_df=min_df,
    )
    model.fit(bundle["item_text"], bundle["item_popularity"])

    preds = model.recommend(
        bundle["eval_users"],
        seen_items_by_user=bundle["seen_items_by_user"],
        k=k,
    )

    metrics = evaluate_predictions(
        bundle["eval_ground_truth"],
        preds,
        catalog_size=bundle["catalog_size"],
        warm_item_ids=bundle["warm_item_ids"],
        cold_item_ids=bundle["cold_item_ids"],
        k=k,
    )
    metrics["model"] = result_name
    metrics["content_weight"] = float(content_weight)
    metrics["popularity_weight"] = float(popularity_weight)
    metrics["content_top_n"] = int(content_top_n)
    metrics["pop_candidates_n"] = int(pop_candidates_n)
    metrics["elapsed_sec"] = round(time.time() - started, 2)

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{result_name}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info("Метрики сохранены: %s", json_path)
    return metrics


# CLI
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Запуск гибридного эксперимента content+popular")
    parser.add_argument("--cache-dir", default="research/results/cache/local_v1")
    parser.add_argument("--results-dir", default="research/results")
    parser.add_argument("--result-name", default="hybrid_content_popular")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--content-weight", type=float, default=0.8)
    parser.add_argument("--popularity-weight", type=float, default=0.2)
    parser.add_argument("--content-top-n", type=int, default=200)
    parser.add_argument("--pop-candidates-n", type=int, default=200)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--min-df", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    metrics = run_hybrid_experiment(
        cache_dir=args.cache_dir,
        results_dir=args.results_dir,
        result_name=args.result_name,
        k=args.k,
        content_weight=args.content_weight,
        popularity_weight=args.popularity_weight,
        content_top_n=args.content_top_n,
        pop_candidates_n=args.pop_candidates_n,
        max_features=args.max_features,
        min_df=args.min_df,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
