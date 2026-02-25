import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from baselines import RandomRecommender, TopPopularRecommender
    from evaluator import evaluate_predictions
    from preprocessor import prepare_research_bundle
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from research.train.baselines import RandomRecommender, TopPopularRecommender  # type: ignore
    from research.train.evaluator import evaluate_predictions  # type: ignore
    from research.train.preprocessor import prepare_research_bundle  # type: ignore


logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# Запустить baseline-эксперименты и сохранить таблицу метрик
def run_baseline_experiments(
    *,
    data_dir: str = "data",
    local_name: str = "local_v1",
    sample_users_n: Optional[int] = None,
    include_random: bool = False,
    k: int = 10,
    results_dir: str = "research/results",
) -> pd.DataFrame:
    bundle = prepare_research_bundle(
        data_dir=data_dir,
        local_name=local_name,
        sample_users_n=sample_users_n,
    )

    rows = []

    logger.info("Обучение TopPopular")
    top_pop = TopPopularRecommender().fit(bundle["local_train"])
    top_preds = top_pop.recommend(
        bundle["eval_users"],
        seen_items_by_user=bundle["seen_items_by_user"],
        k=k,
    )
    top_metrics = evaluate_predictions(
        bundle["eval_ground_truth"],
        top_preds,
        catalog_size=bundle["catalog_size"],
        warm_item_ids=bundle["warm_item_ids"],
        cold_item_ids=bundle["cold_item_ids"],
        k=k,
    )
    top_metrics["model"] = "top_popular"
    rows.append(top_metrics)

    if include_random:
        logger.info("Обучение Random (sanity-check)")
        rnd = RandomRecommender(seed=42).fit(bundle["candidate_item_ids"])
        rnd_preds = rnd.recommend(
            bundle["eval_users"],
            seen_items_by_user=bundle["seen_items_by_user"],
            k=k,
        )
        rnd_metrics = evaluate_predictions(
            bundle["eval_ground_truth"],
            rnd_preds,
            catalog_size=bundle["catalog_size"],
            warm_item_ids=bundle["warm_item_ids"],
            cold_item_ids=bundle["cold_item_ids"],
            k=k,
        )
        rnd_metrics["model"] = "random"
        rows.append(rnd_metrics)

    metrics_df = pd.DataFrame(rows)
    metric_cols = [c for c in metrics_df.columns if c != "model"]
    metrics_df = metrics_df[["model"] + metric_cols]

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "baseline_metrics.csv"
    json_path = out_dir / "baseline_metrics.json"
    metrics_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    logger.info("Сохранены результаты: %s, %s", csv_path, json_path)
    return metrics_df


# CLI
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Запуск baseline экспериментов")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--local-name", default="local_v1")
    parser.add_argument("--sample-users-n", type=int, default=None)
    parser.add_argument("--include-random", action="store_true")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--results-dir", default="research/results")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df = run_baseline_experiments(
        data_dir=args.data_dir,
        local_name=args.local_name,
        sample_users_n=args.sample_users_n,
        include_random=args.include_random,
        k=args.k,
        results_dir=args.results_dir,
    )
    print(df.to_string(index=False))
