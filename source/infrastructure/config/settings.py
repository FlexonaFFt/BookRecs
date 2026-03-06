from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Mapping, MutableMapping, Protocol


@dataclass(frozen=True)
# Описывает настройки приложения.
class Settings:
    pg_dsn: str
    s3_bucket: str
    s3_region: str
    s3_endpoint: str
    train_dataset_dir: str
    train_output_root: str
    train_eval_users_limit: int
    train_candidate_pool_size: int
    train_candidate_per_source_limit: int
    train_pre_top_m: int
    train_final_top_k: int
    train_cf_max_neighbors: int
    train_content_max_neighbors: int
    train_seed: int

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        return cls.from_mapping(environ or os.environ)

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> Settings:
        def _get(name: str, default: str) -> str:
            raw = values.get(name, default)
            return str(raw).strip()

        return cls(
            pg_dsn=_get("BOOKRECS_PG_DSN", ""),
            s3_bucket=_get("BOOKRECS_S3_BUCKET", ""),
            s3_region=_get("BOOKRECS_S3_REGION", "us-east-1"),
            s3_endpoint=_get("BOOKRECS_S3_ENDPOINT", ""),
            train_dataset_dir=_get("BOOKRECS_TRAIN_DATASET_DIR", "artifacts/tmp_preprocessed/goodreads_ya"),
            train_output_root=_get("BOOKRECS_TRAIN_OUTPUT_ROOT", "artifacts/runs"),
            train_eval_users_limit=_parse_positive_int("BOOKRECS_TRAIN_EVAL_USERS_LIMIT", _get, 2000),
            train_candidate_pool_size=_parse_positive_int("BOOKRECS_TRAIN_CANDIDATE_POOL_SIZE", _get, 1000),
            train_candidate_per_source_limit=_parse_positive_int("BOOKRECS_TRAIN_PER_SOURCE_LIMIT", _get, 300),
            train_pre_top_m=_parse_positive_int("BOOKRECS_TRAIN_PRE_TOP_M", _get, 300),
            train_final_top_k=_parse_positive_int("BOOKRECS_TRAIN_FINAL_TOP_K", _get, 10),
            train_cf_max_neighbors=_parse_positive_int("BOOKRECS_TRAIN_CF_MAX_NEIGHBORS", _get, 120),
            train_content_max_neighbors=_parse_positive_int("BOOKRECS_TRAIN_CONTENT_MAX_NEIGHBORS", _get, 120),
            train_seed=_parse_positive_int("BOOKRECS_TRAIN_SEED", _get, 42),
        )

    def to_env_mapping(self) -> dict[str, str]:
        return {
            "BOOKRECS_PG_DSN": self.pg_dsn,
            "BOOKRECS_S3_BUCKET": self.s3_bucket,
            "BOOKRECS_S3_REGION": self.s3_region,
            "BOOKRECS_S3_ENDPOINT": self.s3_endpoint,
            "BOOKRECS_TRAIN_DATASET_DIR": self.train_dataset_dir,
            "BOOKRECS_TRAIN_OUTPUT_ROOT": self.train_output_root,
            "BOOKRECS_TRAIN_EVAL_USERS_LIMIT": str(self.train_eval_users_limit),
            "BOOKRECS_TRAIN_CANDIDATE_POOL_SIZE": str(self.train_candidate_pool_size),
            "BOOKRECS_TRAIN_PER_SOURCE_LIMIT": str(self.train_candidate_per_source_limit),
            "BOOKRECS_TRAIN_PRE_TOP_M": str(self.train_pre_top_m),
            "BOOKRECS_TRAIN_FINAL_TOP_K": str(self.train_final_top_k),
            "BOOKRECS_TRAIN_CF_MAX_NEIGHBORS": str(self.train_cf_max_neighbors),
            "BOOKRECS_TRAIN_CONTENT_MAX_NEIGHBORS": str(self.train_content_max_neighbors),
            "BOOKRECS_TRAIN_SEED": str(self.train_seed),
        }


@dataclass(frozen=True)
class PipelineSettings:
    dataset_name: str
    raw_dir: str
    books_raw_uri: str
    interactions_raw_uri: str
    s3_prefix: str
    k_core: int
    keep_recent_fraction: float
    test_fraction: float
    local_val_fraction: float
    warm_users_only: bool
    language_filter_enabled: bool
    interactions_chunksize: int
    store_backend: str
    registry_backend: str
    s3_bucket: str
    s3_region: str
    s3_endpoint: str
    pg_dsn: str
    migration_path: str
    skip_prepare: bool
    skip_train: bool
    run_migrate: bool
    dataset_dir: str
    output_root: str
    run_name: str | None
    eval_users_limit: int
    candidate_pool_size: int
    candidate_per_source_limit: int
    pre_top_m: int
    final_top_k: int
    cf_mode: str
    cf_max_neighbors: int
    cf_max_items_per_user: int
    content_max_neighbors: int
    seed: int

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> PipelineSettings:
        return cls.from_mapping(environ or os.environ)

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> PipelineSettings:
        core = Settings.from_mapping(values)
        dataset_name = env_str(values, "BOOKRECS_DATASET_NAME", "goodreads_ya")
        raw_dir = env_str(values, "BOOKRECS_RAW_DIR", "data/raw_data")
        cf_mode = env_str(values, "BOOKRECS_TRAIN_CF_MODE", "auto").lower()
        if cf_mode not in {"auto", "fixed"}:
            raise ValueError("BOOKRECS_TRAIN_CF_MODE must be one of: auto, fixed")
        return cls(
            dataset_name=dataset_name,
            raw_dir=raw_dir,
            books_raw_uri=env_str(values, "BOOKRECS_BOOKS_RAW_URI", f"{raw_dir}/books.json.gz"),
            interactions_raw_uri=env_str(values, "BOOKRECS_INTERACTIONS_RAW_URI", f"{raw_dir}/interactions.json.gz"),
            s3_prefix=env_str(values, "BOOKRECS_S3_PREFIX", f"s3://bookrecs/datasets/{dataset_name}"),
            k_core=env_positive_int(values, "BOOKRECS_K_CORE", 2),
            keep_recent_fraction=env_float(values, "BOOKRECS_KEEP_RECENT_FRACTION", 0.6),
            test_fraction=env_float(values, "BOOKRECS_TEST_FRACTION", 0.25),
            local_val_fraction=env_float(values, "BOOKRECS_LOCAL_VAL_FRACTION", 0.2),
            warm_users_only=env_bool(values, "BOOKRECS_WARM_USERS_ONLY", True),
            language_filter_enabled=env_bool(values, "BOOKRECS_LANGUAGE_FILTER_ENABLED", True),
            interactions_chunksize=env_positive_int(values, "BOOKRECS_INTERACTIONS_CHUNKSIZE", 200_000),
            store_backend=env_str(values, "BOOKRECS_STORE_BACKEND", "local"),
            registry_backend=env_str(values, "BOOKRECS_REGISTRY_BACKEND", "memory"),
            s3_bucket=env_str(values, "BOOKRECS_S3_BUCKET", core.s3_bucket),
            s3_region=env_str(values, "BOOKRECS_S3_REGION", core.s3_region),
            s3_endpoint=env_str(values, "BOOKRECS_S3_ENDPOINT", core.s3_endpoint),
            pg_dsn=env_str(values, "BOOKRECS_PG_DSN", core.pg_dsn),
            migration_path=env_str(
                values,
                "BOOKRECS_PG_MIGRATION_PATH",
                "source/infrastructure/storage/postgres/migrations",
            ),
            skip_prepare=env_bool(values, "BOOKRECS_SKIP_PREPARE", False),
            skip_train=env_bool(values, "BOOKRECS_SKIP_TRAIN", False),
            run_migrate=env_bool(values, "BOOKRECS_RUN_MIGRATE", True),
            dataset_dir=env_optional_str(values, "BOOKRECS_TRAIN_DATASET_DIR") or f"artifacts/tmp_preprocessed/{dataset_name}",
            output_root=env_str(values, "BOOKRECS_TRAIN_OUTPUT_ROOT", core.train_output_root),
            run_name=env_optional_str(values, "BOOKRECS_TRAIN_RUN_NAME"),
            eval_users_limit=env_positive_int(values, "BOOKRECS_TRAIN_EVAL_USERS_LIMIT", core.train_eval_users_limit),
            candidate_pool_size=env_positive_int(
                values,
                "BOOKRECS_TRAIN_CANDIDATE_POOL_SIZE",
                core.train_candidate_pool_size,
            ),
            candidate_per_source_limit=env_positive_int(
                values,
                "BOOKRECS_TRAIN_PER_SOURCE_LIMIT",
                core.train_candidate_per_source_limit,
            ),
            pre_top_m=env_positive_int(values, "BOOKRECS_TRAIN_PRE_TOP_M", core.train_pre_top_m),
            final_top_k=env_positive_int(values, "BOOKRECS_TRAIN_FINAL_TOP_K", core.train_final_top_k),
            cf_mode=cf_mode,
            cf_max_neighbors=env_positive_int(values, "BOOKRECS_TRAIN_CF_MAX_NEIGHBORS", core.train_cf_max_neighbors),
            cf_max_items_per_user=env_positive_int(values, "BOOKRECS_TRAIN_CF_MAX_ITEMS_PER_USER", 150),
            content_max_neighbors=env_positive_int(
                values,
                "BOOKRECS_TRAIN_CONTENT_MAX_NEIGHBORS",
                core.train_content_max_neighbors,
            ),
            seed=env_positive_int(values, "BOOKRECS_TRAIN_SEED", core.train_seed),
        )


@dataclass(frozen=True)
class ApiRuntimeSettings:
    model_uri: str
    model_cache_dir: str
    s3_region: str
    s3_endpoint: str
    pg_dsn: str
    history_table: str
    inference_log_table: str
    aws_access_key_id: str | None
    aws_secret_access_key: str | None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ApiRuntimeSettings:
        return cls.from_mapping(environ or os.environ)

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> ApiRuntimeSettings:
        return cls(
            model_uri=env_str(values, "BOOKRECS_API_MODEL_URI", ""),
            model_cache_dir=env_str(values, "BOOKRECS_API_MODEL_CACHE_DIR", "artifacts/cache/models"),
            s3_region=env_str(values, "BOOKRECS_S3_REGION", "us-east-1"),
            s3_endpoint=env_str(values, "BOOKRECS_S3_ENDPOINT", ""),
            pg_dsn=env_str(values, "BOOKRECS_PG_DSN", ""),
            history_table=env_str(values, "BOOKRECS_API_HISTORY_TABLE", "user_item_interactions"),
            inference_log_table=env_str(values, "BOOKRECS_API_INFERENCE_LOG_TABLE", "inference_requests"),
            aws_access_key_id=env_optional_str(values, "AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=env_optional_str(values, "AWS_SECRET_ACCESS_KEY"),
        )


@dataclass(frozen=True)
class ApiServerSettings:
    host: str
    port: int

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ApiServerSettings:
        return cls.from_mapping(environ or os.environ)

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> ApiServerSettings:
        return cls(
            host=env_str(values, "BOOKRECS_API_HOST", "0.0.0.0"),
            port=env_positive_int(values, "BOOKRECS_API_PORT", 8000),
        )


class SettingsReader(Protocol):
    def read(self) -> Settings:
        raise NotImplementedError


class SettingsWriter(Protocol):
    def write(self, settings: Settings) -> None:
        raise NotImplementedError


class EnvSettingsIO(SettingsReader, SettingsWriter):
    def __init__(self, environ: MutableMapping[str, str] | None = None) -> None:
        self._environ = environ if environ is not None else os.environ

    def read(self) -> Settings:
        return Settings.from_env(self._environ)

    def write(self, settings: Settings) -> None:
        self._environ.update(settings.to_env_mapping())


def load_settings() -> Settings:
    return EnvSettingsIO().read()


def save_settings(settings: Settings) -> None:
    EnvSettingsIO().write(settings)


def load_pipeline_settings() -> PipelineSettings:
    return PipelineSettings.from_env()


def load_api_runtime_settings() -> ApiRuntimeSettings:
    return ApiRuntimeSettings.from_env()


def load_api_server_settings() -> ApiServerSettings:
    return ApiServerSettings.from_env()


def env_str(values: Mapping[str, str], name: str, default: str) -> str:
    raw = values.get(name, default)
    return str(raw).strip()


def env_optional_str(values: Mapping[str, str], name: str, default: str | None = None) -> str | None:
    raw = values.get(name)
    if raw is None:
        return default
    value = str(raw).strip()
    return value or default


def env_int(values: Mapping[str, str], name: str, default: int) -> int:
    raw = values.get(name)
    if raw is None:
        return default
    value = str(raw).strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Переменная {name} должна быть целым числом, получено: {raw}") from exc


def env_positive_int(values: Mapping[str, str], name: str, default: int) -> int:
    value = env_int(values, name, default)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def env_float(values: Mapping[str, str], name: str, default: float) -> float:
    raw = values.get(name)
    if raw is None:
        return default
    value = str(raw).strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Переменная {name} должна быть числом, получено: {raw}") from exc


def env_bool(values: Mapping[str, str], name: str, default: bool) -> bool:
    raw = values.get(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Переменная {name} должна быть булевой, получено: {raw}")


def _parse_positive_int(name: str, get_value: Callable[[str, str], str], default: int) -> int:
    raw = get_value(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid int for {name}: {raw}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value
