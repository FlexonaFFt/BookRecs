from __future__ import annotations

from pathlib import Path

from data.goodreads import download_goodreads_raw
from source.application.use_cases import PrepareDataCommand, PrepareDataUseCase
from source.application.use_cases.training import TrainPipelineCommand, TrainPipelineUseCase
from source.domain.entities import DatasetSource, PreprocessingParams
from source.infrastructure.config import load_pipeline_settings
from source.infrastructure.processing.preprocessing import GoodreadsPreprocessor
from source.infrastructure.storage import build_prepare_storage_backends
from source.interfaces.migration_runner import run_migration


# Запускает полный pipeline: migrate -> download -> prepare -> train.
def run_pipeline_from_env() -> None:
    settings = load_pipeline_settings()

    print(f"[pipeline] dataset_name={settings.dataset_name}")
    print(f"[pipeline] raw_dir={settings.raw_dir}")
    print(f"[pipeline] registry_backend={settings.registry_backend} store_backend={settings.store_backend}")

    if settings.run_migrate and settings.registry_backend == "postgres":
        if not settings.pg_dsn.strip():
            raise ValueError("BOOKRECS_PG_DSN is required when BOOKRECS_REGISTRY_BACKEND=postgres")
        run_migration(pg_dsn=settings.pg_dsn, migration_path=settings.migration_path)
        print(f"[pipeline] migrations applied from {settings.migration_path}")

    if not Path(settings.books_raw_uri).exists() or not Path(settings.interactions_raw_uri).exists():
        download_goodreads_raw(raw_dir=settings.raw_dir, force=False)
        print("[pipeline] raw data downloaded")

    if not settings.skip_prepare:
        storage_backends = build_prepare_storage_backends(
            registry_backend=settings.registry_backend,
            pg_dsn=settings.pg_dsn,
            store_backend=settings.store_backend,
            s3_bucket=settings.s3_bucket,
            s3_region=settings.s3_region,
            s3_endpoint=settings.s3_endpoint,
        )
        prepare_use_case = PrepareDataUseCase(
            preprocessor=GoodreadsPreprocessor(),
            dataset_store=storage_backends.dataset_store,
            dataset_registry=storage_backends.dataset_registry,
            run_log=storage_backends.run_log,
        )
        prepare_result = prepare_use_case.execute(
            PrepareDataCommand(
                dataset_name=settings.dataset_name,
                source=DatasetSource(
                    dataset_name=settings.dataset_name,
                    books_raw_uri=settings.books_raw_uri,
                    interactions_raw_uri=settings.interactions_raw_uri,
                ),
                params=PreprocessingParams(
                    k_core=settings.k_core,
                    keep_recent_fraction=settings.keep_recent_fraction,
                    test_fraction=settings.test_fraction,
                    local_val_fraction=settings.local_val_fraction,
                    cold_max_interactions=settings.cold_max_interactions,
                    warm_users_only=settings.warm_users_only,
                    language_filter_enabled=settings.language_filter_enabled,
                    interactions_chunksize=settings.interactions_chunksize,
                ),
                s3_prefix=settings.s3_prefix,
                metadata={"runner": "pipeline_entrypoint"},
            )
        )
        print(f"[pipeline] prepare completed version_id={prepare_result.version_id}")
    else:
        print("[pipeline] prepare skipped")

    if not settings.skip_train:
        train_use_case = TrainPipelineUseCase()
        train_result = train_use_case.execute(
            TrainPipelineCommand(
                dataset_dir=settings.dataset_dir,
                output_root=settings.output_root,
                run_name=settings.run_name,
                eval_users_limit=settings.eval_users_limit,
                cold_max_interactions=settings.cold_max_interactions,
                candidate_pool_size=settings.candidate_pool_size,
                candidate_per_source_limit=settings.candidate_per_source_limit,
                pre_top_m=settings.pre_top_m,
                final_top_k=settings.final_top_k,
                cf_mode=settings.cf_mode,
                cf_max_neighbors=settings.cf_max_neighbors,
                cf_max_items_per_user=settings.cf_max_items_per_user,
                content_max_neighbors=settings.content_max_neighbors,
                seed=settings.seed,
            )
        )
        print(f"[pipeline] train completed run_id={train_result.run_id}")
        print(f"[pipeline] run_dir={train_result.run_dir}")
    else:
        print("[pipeline] train skipped")

def main() -> None:
    run_pipeline_from_env()


if __name__ == "__main__":
    main()
