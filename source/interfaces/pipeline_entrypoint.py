from __future__ import annotations

import json
import os
from pathlib import Path

from data.goodreads import download_goodreads_raw
from source.application.use_cases import PrepareDataCommand, PrepareDataUseCase
from source.application.use_cases.training import (
    TrainPipelineCommand,
    TrainPipelineUseCase,
)
from source.domain.entities import DatasetSource, PipelineRun, PreprocessingParams
from source.infrastructure.config import load_pipeline_settings
from source.infrastructure.inference.model_publisher import upload_model_to_s3
from source.infrastructure.processing.preprocessing import GoodreadsPreprocessor
from source.infrastructure.storage import build_prepare_storage_backends
from source.interfaces.migration_runner import run_migration


# Запускает полный pipeline: migrate -> download -> prepare -> train.
def run_pipeline_from_env() -> None:
    settings = load_pipeline_settings()

    print(f"[pipeline] dataset_name={settings.dataset_name}")
    print(f"[pipeline] raw_dir={settings.raw_dir}")
    print(
        f"[pipeline] registry_backend="
        f"{settings.registry_backend} "
        f"store_backend="
        f"{settings.store_backend}"
    )

    if settings.run_migrate and settings.registry_backend == "postgres":
        if not settings.pg_dsn.strip():
            raise ValueError(
                "BOOKRECS_PG_DSN is required when BOOKRECS_REGISTRY_BACKEND=postgres"
            )
        run_migration(pg_dsn=settings.pg_dsn, migration_path=settings.migration_path)
        print(f"[pipeline] migrations applied from {settings.migration_path}")

    # Always build storage_backends so run_log is available for the train step too
    storage_backends = build_prepare_storage_backends(
        registry_backend=settings.registry_backend,
        pg_dsn=settings.pg_dsn,
        store_backend=settings.store_backend,
        s3_bucket=settings.s3_bucket,
        s3_region=settings.s3_region,
        s3_endpoint=settings.s3_endpoint,
        s3_verify_ssl=settings.s3_verify_ssl,
    )

    if not settings.skip_prepare:

        def _ensure_raw_data() -> None:
            if (
                not Path(settings.books_raw_uri).exists()
                or not Path(settings.interactions_raw_uri).exists()
            ):
                download_goodreads_raw(raw_dir=settings.raw_dir, force=False)
                print("[pipeline] raw data downloaded")

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
                    max_users=settings.max_users,
                    max_interactions_rows=settings.max_interactions_rows,
                ),
                s3_prefix=settings.s3_prefix,
                metadata={"runner": "pipeline_entrypoint"},
                local_dataset_dir=settings.dataset_dir,
                ensure_raw_data_fn=_ensure_raw_data,
            )
        )
        print(f"[pipeline] prepare completed version_id={prepare_result.version_id}")
    else:
        print("[pipeline] prepare skipped")

    if not settings.skip_train:
        train_use_case = TrainPipelineUseCase()
        train_run = PipelineRun(
            run_id=settings.run_name or "train",
            pipeline_name="bookrecs",
        )
        storage_backends.run_log.start(train_run)
        try:
            train_result = train_use_case.execute(
                TrainPipelineCommand(
                    dataset_dir=settings.dataset_dir,
                    output_root=settings.output_root,
                    run_name=settings.run_name,
                    train_profile=settings.train_profile,
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
                    prerank_model=settings.prerank_model,
                    catboost_iterations=settings.catboost_iterations,
                    catboost_depth=settings.catboost_depth,
                    catboost_learning_rate=settings.catboost_learning_rate,
                    seed=settings.seed,
                    data_fraction=settings.train_data_fraction,
                )
            )
        except Exception as exc:
            train_run.mark_failed(str(exc))
            storage_backends.run_log.finish(train_run)
            raise

        metrics: dict[str, float] = {}
        try:
            raw = json.loads(
                Path(train_result.metrics_path).read_text(encoding="utf-8")
            )
            metrics = {
                k: float(v) for k, v in raw.items() if isinstance(v, (int, float))
            }
        except Exception as exc:
            print(f"[pipeline] warning: could not read metrics file: {exc}", flush=True)

        train_run.run_id = train_result.run_id
        train_run.metrics = metrics
        train_run.mark_success()
        storage_backends.run_log.finish(train_run)

        _publish_train_artifacts_to_s3_if_enabled(
            run_id=train_result.run_id,
            output_root=settings.output_root,
            store_backend=settings.store_backend,
            s3_bucket=settings.s3_bucket,
            s3_region=settings.s3_region,
            s3_endpoint=settings.s3_endpoint,
            s3_verify_ssl=settings.s3_verify_ssl,
        )

        print(f"[pipeline] train completed run_id={train_result.run_id}")
        print(f"[pipeline] metrics={metrics}")
        print(f"[pipeline] run_dir={train_result.run_dir}")
        print("[pipeline] promotion is handled by a separate quality gate", flush=True)
    else:
        print("[pipeline] train skipped")


def main() -> None:
    run_pipeline_from_env()


def _publish_train_artifacts_to_s3_if_enabled(
    *,
    run_id: str,
    output_root: str,
    store_backend: str,
    s3_bucket: str,
    s3_region: str,
    s3_endpoint: str,
    s3_verify_ssl: bool,
) -> None:
    if store_backend.strip().lower() != "s3":
        return
    upload_enabled = _env_bool(
        "BOOKRECS_TRAIN_UPLOAD_MODEL_ARTIFACTS",
        _env_bool("BOOKRECS_TRAIN_UPLOAD_RUN_ARTIFACTS", True),
    )
    if not upload_enabled:
        print("[pipeline] skip S3 model upload (disabled by env)", flush=True)
        return
    if not s3_bucket.strip():
        raise ValueError("BOOKRECS_S3_BUCKET is required for S3 model upload")

    models_prefix = (
        os.getenv("BOOKRECS_TRAIN_S3_MODELS_PREFIX") or ""
    ).strip() or "models"
    s3_uri = upload_model_to_s3(
        run_id=run_id,
        output_root=output_root,
        bucket=s3_bucket,
        s3_prefix=models_prefix,
        s3_region=s3_region,
        s3_endpoint=s3_endpoint or None,
        verify_ssl=s3_verify_ssl,
    )
    print(f"[pipeline] model s3_uri={s3_uri}", flush=True)


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on", "y"}


if __name__ == "__main__":
    main()
