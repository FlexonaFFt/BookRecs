import argparse
import logging
from pathlib import Path
import time

try:
    from preprocessor import prepare_research_bundle, save_research_bundle_cache
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from research.train.preprocessor import prepare_research_bundle, save_research_bundle_cache  # type: ignore


logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# Подготовить и сохранить общий кэш данных для research
def run_precompute(
    data_dir: str = "data",
    local_name: str = "local_v1",
    sample_users_n: int = None,
    cache_dir: str = "research/results/cache/local_v1",
    seed: int = 42,
) -> None:
    started = time.time()
    logger.info("Старт precompute research bundle")
    bundle = prepare_research_bundle(
        data_dir=data_dir,
        local_name=local_name,
        sample_users_n=sample_users_n,
        seed=seed,
    )
    save_research_bundle_cache(bundle, cache_dir=cache_dir)
    logger.info("Precompute завершен за %.1f сек", time.time() - started)


# CLI
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Подготовка кэша данных для research")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--local-name", default="local_v1")
    parser.add_argument("--sample-users-n", type=int, default=None)
    parser.add_argument("--cache-dir", default="research/results/cache/local_v1")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_precompute(
        data_dir=args.data_dir,
        local_name=args.local_name,
        sample_users_n=args.sample_users_n,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
