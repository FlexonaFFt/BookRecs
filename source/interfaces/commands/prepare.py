from __future__ import annotations

import argparse

from source.application.use_cases import PrepareDataCommand, PrepareDataUseCase
from source.domain.entities import DatasetSource, PreprocessingParams
from source.infrastructure.config import load_settings
from source.infrastructure.processing.preprocessing import GoodreadsPreprocessor
from source.infrastructure.storage import build_prepare_storage_backends
from source.interfaces.commands.env_defaults import env_bool, env_float, env_int, env_str


def build_parser() -> argparse.ArgumentParser:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Prepare dataset pipeline")
    dataset_name = env_str("BOOKRECS_DATASET_NAME", "goodreads_ya")
    raw_dir = env_str("BOOKRECS_RAW_DIR", "data/raw_data")
    parser.add_argument("--dataset-name", default=dataset_name)
    parser.add_argument("--books-raw-uri", default=env_str("BOOKRECS_BOOKS_RAW_URI", f"{raw_dir}/books.json.gz"))
    parser.add_argument(
        "--interactions-raw-uri",
        default=env_str("BOOKRECS_INTERACTIONS_RAW_URI", f"{raw_dir}/interactions.json.gz"),
    )
    parser.add_argument("--s3-prefix", default=env_str("BOOKRECS_S3_PREFIX", f"s3://bookrecs/datasets/{dataset_name}"))
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
    parser.add_argument(
        "--registry-backend",
        choices=["memory", "postgres"],
        default=env_str("BOOKRECS_REGISTRY_BACKEND", "memory"),
    )
    parser.add_argument("--pg-dsn", default=settings.pg_dsn)
    parser.add_argument("--store-backend", choices=["local", "s3"], default=env_str("BOOKRECS_STORE_BACKEND", "local"))
    parser.add_argument("--s3-bucket", default=settings.s3_bucket)
    parser.add_argument("--s3-region", default=settings.s3_region)
    parser.add_argument("--s3-endpoint", default=settings.s3_endpoint)
    return parser


def run_prepare(args: argparse.Namespace) -> None:
    source = DatasetSource(
        dataset_name=args.dataset_name,
        books_raw_uri=args.books_raw_uri,
        interactions_raw_uri=args.interactions_raw_uri,
    )
    params = PreprocessingParams(
        k_core=args.k_core,
        keep_recent_fraction=args.keep_recent_fraction,
        test_fraction=args.test_fraction,
        local_val_fraction=args.local_val_fraction,
        warm_users_only=args.warm_users_only,
        language_filter_enabled=args.language_filter_enabled,
        interactions_chunksize=args.interactions_chunksize,
    )

    storage_backends = build_prepare_storage_backends(
        registry_backend=args.registry_backend,
        pg_dsn=args.pg_dsn,
        store_backend=args.store_backend,
        s3_bucket=args.s3_bucket,
        s3_region=args.s3_region,
        s3_endpoint=args.s3_endpoint,
    )

    use_case = PrepareDataUseCase(
        preprocessor=GoodreadsPreprocessor(),
        dataset_store=storage_backends.dataset_store,
        dataset_registry=storage_backends.dataset_registry,
        run_log=storage_backends.run_log,
    )

    result = use_case.execute(
        PrepareDataCommand(
            dataset_name=args.dataset_name,
            source=source,
            params=params,
            s3_prefix=args.s3_prefix,
            metadata={"runner": "local_cli"},
        )
    )

    print("prepare-data completed")
    print(f"dataset_name={result.dataset_name}")
    print(f"version_id={result.version_id}")
    print(f"params_hash={result.params_hash}")
    print(f"s3_prefix={result.s3_prefix}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_prepare(args)


if __name__ == "__main__":
    main()
