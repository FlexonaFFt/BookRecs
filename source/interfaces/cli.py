from __future__ import annotations

import argparse
import os

from source.application.use_cases import PrepareDataCommand, PrepareDataUseCase
from source.domain.entities import DatasetSource, PreprocessingParams
from source.infrastructure.preprocessing import PreprocessorStyle
from source.infrastructure.storage import (
    ClientPg,
    RegistryMemory,
    RegistryPg,
    RunLogMemory,
    RunLogPg,
    StoreLocal,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BookRecs CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare-data", help="Run preprocessing pipeline")
    prep.add_argument("--dataset-name", default="goodreads_ya")
    prep.add_argument("--books-raw-uri", default="data/raw_data/books.json.gz")
    prep.add_argument("--interactions-raw-uri", default="data/raw_data/interactions.json.gz")
    prep.add_argument("--s3-prefix", default="s3://bookrecs/datasets/goodreads_ya")
    prep.add_argument("--k-core", type=int, default=2)
    prep.add_argument("--keep-recent-fraction", type=float, default=0.6)
    prep.add_argument("--test-fraction", type=float, default=0.25)
    prep.add_argument("--local-val-fraction", type=float, default=0.2)
    prep.add_argument("--warm-users-only", action=argparse.BooleanOptionalAction, default=True)
    prep.add_argument("--language-filter-enabled", action=argparse.BooleanOptionalAction, default=True)
    prep.add_argument("--interactions-chunksize", type=int, default=200_000)
    prep.add_argument("--registry-backend", choices=["memory", "postgres"], default="memory")
    prep.add_argument("--pg-dsn", default=os.getenv("BOOKRECS_PG_DSN", ""))

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command != "prepare-data":
        raise ValueError(f"Unknown command: {args.command}")

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

    if args.registry_backend == "postgres":
        if not args.pg_dsn.strip():
            raise ValueError("pg-dsn is required when --registry-backend=postgres")
        pg = ClientPg(args.pg_dsn)
        dataset_registry = RegistryPg(pg)
        run_log = RunLogPg(pg)
    else:
        dataset_registry = RegistryMemory()
        run_log = RunLogMemory()

    use_case = PrepareDataUseCase(
        preprocessor=PreprocessorStyle(),
        dataset_store=StoreLocal(),
        dataset_registry=dataset_registry,
        run_log=run_log,
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


if __name__ == "__main__":
    main()
