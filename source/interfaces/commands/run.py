from __future__ import annotations

import argparse
from pathlib import Path

from data.goodreads import download_goodreads_raw
from source.infrastructure.config import load_settings
from source.interfaces.commands.env_defaults import env_bool, env_float, env_int, env_optional_str, env_str
from source.interfaces.commands.migrate import run_migration
from source.interfaces.commands.prepare import build_parser as build_prepare_parser
from source.interfaces.commands.prepare import run_prepare
from source.interfaces.commands.train import build_parser as build_train_parser
from source.interfaces.commands.train import run_train


def build_parser() -> argparse.ArgumentParser:
    settings = load_settings()
    dataset_name = env_str("BOOKRECS_DATASET_NAME", "goodreads_ya")
    raw_dir = env_str("BOOKRECS_RAW_DIR", "data/raw_data")
    parser = argparse.ArgumentParser(description="Run full pipeline: download -> prepare -> train")
    parser.add_argument("--dataset-name", default=dataset_name)
    parser.add_argument("--raw-dir", default=raw_dir)
    parser.add_argument("--books-raw-uri", default=env_optional_str("BOOKRECS_BOOKS_RAW_URI"))
    parser.add_argument("--interactions-raw-uri", default=env_optional_str("BOOKRECS_INTERACTIONS_RAW_URI"))
    parser.add_argument("--dataset-dir", default=env_optional_str("BOOKRECS_TRAIN_DATASET_DIR"))
    parser.add_argument("--output-root", default=settings.train_output_root)
    parser.add_argument("--run-name", default=env_optional_str("BOOKRECS_TRAIN_RUN_NAME"))
    parser.add_argument("--k-core", type=int, default=env_int("BOOKRECS_K_CORE", 2))
    parser.add_argument("--keep-recent-fraction", type=float, default=env_float("BOOKRECS_KEEP_RECENT_FRACTION", 0.6))
    parser.add_argument("--test-fraction", type=float, default=env_float("BOOKRECS_TEST_FRACTION", 0.25))
    parser.add_argument("--local-val-fraction", type=float, default=env_float("BOOKRECS_LOCAL_VAL_FRACTION", 0.2))
    parser.add_argument(
        "--warm-users-only",
        action=argparse.BooleanOptionalAction,
        default=env_bool("BOOKRECS_WARM_USERS_ONLY", True),
    )
    parser.add_argument(
        "--language-filter-enabled",
        action=argparse.BooleanOptionalAction,
        default=env_bool("BOOKRECS_LANGUAGE_FILTER_ENABLED", True),
    )
    parser.add_argument("--interactions-chunksize", type=int, default=env_int("BOOKRECS_INTERACTIONS_CHUNKSIZE", 200_000))
    parser.add_argument("--store-backend", choices=["local", "s3"], default=env_str("BOOKRECS_STORE_BACKEND", "local"))
    parser.add_argument(
        "--registry-backend",
        choices=["memory", "postgres"],
        default=env_str("BOOKRECS_REGISTRY_BACKEND", "memory"),
    )
    parser.add_argument("--s3-prefix", default=env_optional_str("BOOKRECS_S3_PREFIX"))
    parser.add_argument("--s3-bucket", default=settings.s3_bucket)
    parser.add_argument("--s3-region", default=settings.s3_region)
    parser.add_argument("--s3-endpoint", default=settings.s3_endpoint)
    parser.add_argument("--pg-dsn", default=settings.pg_dsn)
    parser.add_argument(
        "--migration-path",
        default=env_str("BOOKRECS_PG_MIGRATION_PATH", "source/infrastructure/storage/postgres/migrations"),
    )
    parser.add_argument("--skip-prepare", action=argparse.BooleanOptionalAction, default=env_bool("BOOKRECS_SKIP_PREPARE", False))
    parser.add_argument("--skip-train", action=argparse.BooleanOptionalAction, default=env_bool("BOOKRECS_SKIP_TRAIN", False))
    parser.add_argument("--migrate", action=argparse.BooleanOptionalAction, default=env_bool("BOOKRECS_RUN_MIGRATE", True))
    parser.add_argument("--eval-users-limit", type=int, default=env_int("BOOKRECS_TRAIN_EVAL_USERS_LIMIT", settings.train_eval_users_limit))
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=env_int("BOOKRECS_TRAIN_CANDIDATE_POOL_SIZE", settings.train_candidate_pool_size),
    )
    parser.add_argument(
        "--candidate-per-source-limit",
        type=int,
        default=env_int("BOOKRECS_TRAIN_PER_SOURCE_LIMIT", settings.train_candidate_per_source_limit),
    )
    parser.add_argument("--pre-top-m", type=int, default=env_int("BOOKRECS_TRAIN_PRE_TOP_M", settings.train_pre_top_m))
    parser.add_argument("--final-top-k", type=int, default=env_int("BOOKRECS_TRAIN_FINAL_TOP_K", settings.train_final_top_k))
    parser.add_argument(
        "--cf-max-neighbors",
        type=int,
        default=env_int("BOOKRECS_TRAIN_CF_MAX_NEIGHBORS", settings.train_cf_max_neighbors),
    )
    parser.add_argument(
        "--content-max-neighbors",
        type=int,
        default=env_int("BOOKRECS_TRAIN_CONTENT_MAX_NEIGHBORS", settings.train_content_max_neighbors),
    )
    parser.add_argument("--seed", type=int, default=env_int("BOOKRECS_TRAIN_SEED", settings.train_seed))
    return parser


def run_pipeline(args: argparse.Namespace) -> None:
    books_raw_uri = args.books_raw_uri or f"{args.raw_dir}/books.json.gz"
    interactions_raw_uri = args.interactions_raw_uri or f"{args.raw_dir}/interactions.json.gz"
    s3_prefix = args.s3_prefix or f"s3://bookrecs/datasets/{args.dataset_name}"
    dataset_dir = args.dataset_dir or f"artifacts/tmp_preprocessed/{args.dataset_name}"

    if args.migrate and args.registry_backend == "postgres":
        if not args.pg_dsn.strip():
            raise ValueError("--pg-dsn is required when --registry-backend=postgres")
        run_migration(pg_dsn=args.pg_dsn, migration_path=args.migration_path)

    if not Path(books_raw_uri).exists() or not Path(interactions_raw_uri).exists():
        download_goodreads_raw(raw_dir=args.raw_dir, force=False)

    if not args.skip_prepare:
        prepare_args = build_prepare_parser().parse_args([])
        prepare_args.dataset_name = args.dataset_name
        prepare_args.books_raw_uri = books_raw_uri
        prepare_args.interactions_raw_uri = interactions_raw_uri
        prepare_args.s3_prefix = s3_prefix
        prepare_args.registry_backend = args.registry_backend
        prepare_args.pg_dsn = args.pg_dsn
        prepare_args.store_backend = args.store_backend
        prepare_args.s3_bucket = args.s3_bucket
        prepare_args.s3_region = args.s3_region
        prepare_args.s3_endpoint = args.s3_endpoint
        prepare_args.k_core = args.k_core
        prepare_args.keep_recent_fraction = args.keep_recent_fraction
        prepare_args.test_fraction = args.test_fraction
        prepare_args.local_val_fraction = args.local_val_fraction
        prepare_args.warm_users_only = args.warm_users_only
        prepare_args.language_filter_enabled = args.language_filter_enabled
        prepare_args.interactions_chunksize = args.interactions_chunksize
        run_prepare(prepare_args)

    if not args.skip_train:
        train_args = build_train_parser().parse_args([])
        train_args.dataset_dir = dataset_dir
        train_args.output_root = args.output_root
        train_args.run_name = args.run_name
        train_args.eval_users_limit = args.eval_users_limit
        train_args.candidate_pool_size = args.candidate_pool_size
        train_args.candidate_per_source_limit = args.candidate_per_source_limit
        train_args.pre_top_m = args.pre_top_m
        train_args.final_top_k = args.final_top_k
        train_args.cf_max_neighbors = args.cf_max_neighbors
        train_args.content_max_neighbors = args.content_max_neighbors
        train_args.seed = args.seed
        run_train(train_args)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
