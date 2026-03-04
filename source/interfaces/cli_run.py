from __future__ import annotations

import argparse
from pathlib import Path

from data.goodreads import download_goodreads_raw
from source.infrastructure.config import load_settings
from source.interfaces.cli_migrate import run_migration
from source.interfaces.cli_prepare import build_parser as build_prepare_parser
from source.interfaces.cli_prepare import run_prepare
from source.interfaces.cli_train import build_parser as build_train_parser
from source.interfaces.cli_train import run_train


def build_parser() -> argparse.ArgumentParser:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Run full pipeline: download -> prepare -> train")
    parser.add_argument("--dataset-name", default="goodreads_ya")
    parser.add_argument("--raw-dir", default="data/raw_data")
    parser.add_argument("--books-raw-uri", default=None)
    parser.add_argument("--interactions-raw-uri", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--output-root", default=settings.train_output_root)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--k-core", type=int, default=2)
    parser.add_argument("--keep-recent-fraction", type=float, default=0.6)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--local-val-fraction", type=float, default=0.2)
    parser.add_argument("--warm-users-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--language-filter-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interactions-chunksize", type=int, default=200_000)
    parser.add_argument("--store-backend", choices=["local", "s3"], default="local")
    parser.add_argument("--registry-backend", choices=["memory", "postgres"], default="memory")
    parser.add_argument("--s3-prefix", default=None)
    parser.add_argument("--s3-bucket", default=settings.s3_bucket)
    parser.add_argument("--s3-region", default=settings.s3_region)
    parser.add_argument("--s3-endpoint", default=settings.s3_endpoint)
    parser.add_argument("--pg-dsn", default=settings.pg_dsn)
    parser.add_argument(
        "--migration-path",
        default="source/infrastructure/storage/postgres/migrations",
    )
    parser.add_argument("--skip-prepare", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--migrate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-users-limit", type=int, default=settings.train_eval_users_limit)
    parser.add_argument("--candidate-pool-size", type=int, default=settings.train_candidate_pool_size)
    parser.add_argument("--candidate-per-source-limit", type=int, default=settings.train_candidate_per_source_limit)
    parser.add_argument("--pre-top-m", type=int, default=settings.train_pre_top_m)
    parser.add_argument("--final-top-k", type=int, default=settings.train_final_top_k)
    parser.add_argument("--cf-max-neighbors", type=int, default=settings.train_cf_max_neighbors)
    parser.add_argument("--content-max-neighbors", type=int, default=settings.train_content_max_neighbors)
    parser.add_argument("--seed", type=int, default=settings.train_seed)
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
