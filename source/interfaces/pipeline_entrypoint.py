from __future__ import annotations

import os
from pathlib import Path

from data.goodreads import download_goodreads_raw
from source.application.use_cases import PrepareDataCommand, PrepareDataUseCase
from source.application.use_cases.training import TrainPipelineCommand, TrainPipelineUseCase
from source.domain.entities import DatasetSource, PreprocessingParams
from source.infrastructure.config import load_settings
from source.infrastructure.processing.preprocessing import GoodreadsPreprocessor
from source.infrastructure.storage import build_prepare_storage_backends
from source.interfaces.migration_runner import run_migration


TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}


# Запускает полный pipeline: migrate -> download -> prepare -> train.
def run_pipeline_from_env() -> None:
    settings = load_settings()

    # Базовые идентификаторы датасета и пути к raw-источникам.
    dataset_name = _env_str("BOOKRECS_DATASET_NAME", "goodreads_ya")
    raw_dir = _env_str("BOOKRECS_RAW_DIR", "data/raw_data")
    books_raw_uri = _env_str("BOOKRECS_BOOKS_RAW_URI", f"{raw_dir}/books.json.gz")
    interactions_raw_uri = _env_str("BOOKRECS_INTERACTIONS_RAW_URI", f"{raw_dir}/interactions.json.gz")
    s3_prefix = _env_str("BOOKRECS_S3_PREFIX", f"s3://bookrecs/datasets/{dataset_name}")

    # Параметры предобработки данных перед обучением.
    k_core = _env_int("BOOKRECS_K_CORE", 2)
    keep_recent_fraction = _env_float("BOOKRECS_KEEP_RECENT_FRACTION", 0.6)
    test_fraction = _env_float("BOOKRECS_TEST_FRACTION", 0.25)
    local_val_fraction = _env_float("BOOKRECS_LOCAL_VAL_FRACTION", 0.2)
    warm_users_only = _env_bool("BOOKRECS_WARM_USERS_ONLY", True)
    language_filter_enabled = _env_bool("BOOKRECS_LANGUAGE_FILTER_ENABLED", True)
    interactions_chunksize = _env_int("BOOKRECS_INTERACTIONS_CHUNKSIZE", 200_000)

    # Выбор backend-ов хранения и параметры подключений.
    store_backend = _env_str("BOOKRECS_STORE_BACKEND", "local")
    registry_backend = _env_str("BOOKRECS_REGISTRY_BACKEND", "memory")
    s3_bucket = _env_str("BOOKRECS_S3_BUCKET", settings.s3_bucket)
    s3_region = _env_str("BOOKRECS_S3_REGION", settings.s3_region)
    s3_endpoint = _env_str("BOOKRECS_S3_ENDPOINT", settings.s3_endpoint)
    pg_dsn = _env_str("BOOKRECS_PG_DSN", settings.pg_dsn)
    migration_path = _env_str("BOOKRECS_PG_MIGRATION_PATH", "source/infrastructure/storage/postgres/migrations")

    # Флаги оркестрации: какие этапы выполнять в этом запуске.
    skip_prepare = _env_bool("BOOKRECS_SKIP_PREPARE", False)
    skip_train = _env_bool("BOOKRECS_SKIP_TRAIN", False)
    run_migrate = _env_bool("BOOKRECS_RUN_MIGRATE", True)

    # Параметры этапа обучения и пути вывода артефактов.
    dataset_dir = _env_optional_str("BOOKRECS_TRAIN_DATASET_DIR") or f"artifacts/tmp_preprocessed/{dataset_name}"
    output_root = _env_str("BOOKRECS_TRAIN_OUTPUT_ROOT", settings.train_output_root)
    run_name = _env_optional_str("BOOKRECS_TRAIN_RUN_NAME")
    eval_users_limit = _env_int("BOOKRECS_TRAIN_EVAL_USERS_LIMIT", settings.train_eval_users_limit)
    candidate_pool_size = _env_int("BOOKRECS_TRAIN_CANDIDATE_POOL_SIZE", settings.train_candidate_pool_size)
    candidate_per_source_limit = _env_int("BOOKRECS_TRAIN_PER_SOURCE_LIMIT", settings.train_candidate_per_source_limit)
    pre_top_m = _env_int("BOOKRECS_TRAIN_PRE_TOP_M", settings.train_pre_top_m)
    final_top_k = _env_int("BOOKRECS_TRAIN_FINAL_TOP_K", settings.train_final_top_k)
    cf_mode = _env_str("BOOKRECS_TRAIN_CF_MODE", "auto").strip().lower()
    if cf_mode not in {"auto", "fixed"}:
        raise ValueError("BOOKRECS_TRAIN_CF_MODE must be one of: auto, fixed")
    cf_max_neighbors = _env_int("BOOKRECS_TRAIN_CF_MAX_NEIGHBORS", settings.train_cf_max_neighbors)
    cf_max_items_per_user = _env_int("BOOKRECS_TRAIN_CF_MAX_ITEMS_PER_USER", 150)
    content_max_neighbors = _env_int("BOOKRECS_TRAIN_CONTENT_MAX_NEIGHBORS", settings.train_content_max_neighbors)
    seed = _env_int("BOOKRECS_TRAIN_SEED", settings.train_seed)

    print(f"[pipeline] dataset_name={dataset_name}")
    print(f"[pipeline] raw_dir={raw_dir}")
    print(f"[pipeline] registry_backend={registry_backend} store_backend={store_backend}")

    if run_migrate and registry_backend == "postgres":
        if not pg_dsn.strip():
            raise ValueError("BOOKRECS_PG_DSN is required when BOOKRECS_REGISTRY_BACKEND=postgres")
        run_migration(pg_dsn=pg_dsn, migration_path=migration_path)
        print(f"[pipeline] migrations applied from {migration_path}")

    if not Path(books_raw_uri).exists() or not Path(interactions_raw_uri).exists():
        download_goodreads_raw(raw_dir=raw_dir, force=False)
        print("[pipeline] raw data downloaded")

    if not skip_prepare:
        storage_backends = build_prepare_storage_backends(
            registry_backend=registry_backend,
            pg_dsn=pg_dsn,
            store_backend=store_backend,
            s3_bucket=s3_bucket,
            s3_region=s3_region,
            s3_endpoint=s3_endpoint,
        )
        prepare_use_case = PrepareDataUseCase(
            preprocessor=GoodreadsPreprocessor(),
            dataset_store=storage_backends.dataset_store,
            dataset_registry=storage_backends.dataset_registry,
            run_log=storage_backends.run_log,
        )
        prepare_result = prepare_use_case.execute(
            PrepareDataCommand(
                dataset_name=dataset_name,
                source=DatasetSource(
                    dataset_name=dataset_name,
                    books_raw_uri=books_raw_uri,
                    interactions_raw_uri=interactions_raw_uri,
                ),
                params=PreprocessingParams(
                    k_core=k_core,
                    keep_recent_fraction=keep_recent_fraction,
                    test_fraction=test_fraction,
                    local_val_fraction=local_val_fraction,
                    warm_users_only=warm_users_only,
                    language_filter_enabled=language_filter_enabled,
                    interactions_chunksize=interactions_chunksize,
                ),
                s3_prefix=s3_prefix,
                metadata={"runner": "pipeline_entrypoint"},
            )
        )
        print(f"[pipeline] prepare completed version_id={prepare_result.version_id}")
    else:
        print("[pipeline] prepare skipped")

    if not skip_train:
        train_use_case = TrainPipelineUseCase()
        train_result = train_use_case.execute(
            TrainPipelineCommand(
                dataset_dir=dataset_dir,
                output_root=output_root,
                run_name=run_name,
                eval_users_limit=eval_users_limit,
                candidate_pool_size=candidate_pool_size,
                candidate_per_source_limit=candidate_per_source_limit,
                pre_top_m=pre_top_m,
                final_top_k=final_top_k,
                cf_mode=cf_mode,
                cf_max_neighbors=cf_max_neighbors,
                cf_max_items_per_user=cf_max_items_per_user,
                content_max_neighbors=content_max_neighbors,
                seed=seed,
            )
        )
        print(f"[pipeline] train completed run_id={train_result.run_id}")
        print(f"[pipeline] run_dir={train_result.run_dir}")
    else:
        print("[pipeline] train skipped")


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_optional_str(name: str, default: str | None = None) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Переменная {name} должна быть целым числом, получено: {raw}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Переменная {name} должна быть числом, получено: {raw}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if not value:
        return default
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    raise ValueError(f"Переменная {name} должна быть булевой, получено: {raw}")


def main() -> None:
    run_pipeline_from_env()


if __name__ == "__main__":
    main()
