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
    train_profile: str
    train_auto_tune: bool
    train_eval_users_limit: int
    cold_max_interactions: int
    train_candidate_pool_size: int
    train_candidate_per_source_limit: int
    train_pre_top_m: int
    train_final_top_k: int
    train_cf_max_neighbors: int
    train_cf_max_items_per_user: int
    train_content_max_neighbors: int
    train_prerank_model: str
    train_catboost_iterations: int
    train_catboost_depth: int
    train_catboost_learning_rate: float
    train_seed: int

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        return cls.from_mapping(environ or os.environ)

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> Settings:
        def _get(name: str, default: str) -> str:
            raw = values.get(name, default)
            return str(raw).strip()

        profile = _resolve_train_profile(_get)
        if profile not in {"auto", "default", "lite"}:
            raise ValueError(
                "BOOKRECS_TRAIN_PROFILE must be one of: auto, default, lite"
            )

        auto_tune = env_bool(values, "BOOKRECS_TRAIN_AUTO_TUNE", False)
        profile_defaults = _profile_defaults(profile)
        prerank_model = _resolve_train_value(
            values,
            "BOOKRECS_TRAIN_PRERANK_MODEL",
            str(profile_defaults["prerank_model"]),
            auto_tune=auto_tune,
        ).lower()
        if prerank_model not in {"auto", "catboost", "linear"}:
            raise ValueError(
                "BOOKRECS_TRAIN_PRERANK_MODEL must be one of: auto, catboost, linear"
            )

        return cls(
            pg_dsn=_get("BOOKRECS_PG_DSN", ""),
            s3_bucket=_get("BOOKRECS_S3_BUCKET", ""),
            s3_region=_get("BOOKRECS_S3_REGION", "us-east-1"),
            s3_endpoint=_get("BOOKRECS_S3_ENDPOINT", ""),
            train_dataset_dir=_get(
                "BOOKRECS_TRAIN_DATASET_DIR", "artifacts/tmp_preprocessed/goodreads_ya"
            ),
            train_output_root=_get("BOOKRECS_TRAIN_OUTPUT_ROOT", "artifacts/runs"),
            train_profile=profile,
            train_auto_tune=auto_tune,
            train_eval_users_limit=_read_positive_int(
                values,
                "BOOKRECS_TRAIN_EVAL_USERS_LIMIT",
                int(profile_defaults["eval_users_limit"]),
                auto_tune=auto_tune,
            ),
            cold_max_interactions=_parse_non_negative_int(
                "BOOKRECS_COLD_MAX_INTERACTIONS", _get, 5
            ),
            train_candidate_pool_size=_read_positive_int(
                values,
                "BOOKRECS_TRAIN_CANDIDATE_POOL_SIZE",
                int(profile_defaults["candidate_pool_size"]),
                auto_tune=auto_tune,
            ),
            train_candidate_per_source_limit=_read_positive_int(
                values,
                "BOOKRECS_TRAIN_PER_SOURCE_LIMIT",
                int(profile_defaults["candidate_per_source_limit"]),
                auto_tune=auto_tune,
            ),
            train_pre_top_m=_read_positive_int(
                values,
                "BOOKRECS_TRAIN_PRE_TOP_M",
                int(profile_defaults["pre_top_m"]),
                auto_tune=auto_tune,
            ),
            train_final_top_k=_parse_positive_int(
                "BOOKRECS_TRAIN_FINAL_TOP_K", _get, 10
            ),
            train_cf_max_neighbors=_read_positive_int(
                values,
                "BOOKRECS_TRAIN_CF_MAX_NEIGHBORS",
                int(profile_defaults["cf_max_neighbors"]),
                auto_tune=auto_tune,
            ),
            train_cf_max_items_per_user=_read_positive_int(
                values,
                "BOOKRECS_TRAIN_CF_MAX_ITEMS_PER_USER",
                int(profile_defaults["cf_max_items_per_user"]),
                auto_tune=auto_tune,
            ),
            train_content_max_neighbors=_read_positive_int(
                values,
                "BOOKRECS_TRAIN_CONTENT_MAX_NEIGHBORS",
                int(profile_defaults["content_max_neighbors"]),
                auto_tune=auto_tune,
            ),
            train_prerank_model=prerank_model,
            train_catboost_iterations=_read_positive_int(
                values,
                "BOOKRECS_TRAIN_CATBOOST_ITERATIONS",
                int(profile_defaults["catboost_iterations"]),
                auto_tune=auto_tune,
            ),
            train_catboost_depth=_read_positive_int(
                values,
                "BOOKRECS_TRAIN_CATBOOST_DEPTH",
                int(profile_defaults["catboost_depth"]),
                auto_tune=auto_tune,
            ),
            train_catboost_learning_rate=_read_positive_float(
                values,
                "BOOKRECS_TRAIN_CATBOOST_LEARNING_RATE",
                float(profile_defaults["catboost_learning_rate"]),
                auto_tune=auto_tune,
            ),
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
            "BOOKRECS_TRAIN_PROFILE": self.train_profile,
            "BOOKRECS_TRAIN_AUTO_TUNE": "true" if self.train_auto_tune else "false",
            "BOOKRECS_TRAIN_EVAL_USERS_LIMIT": str(self.train_eval_users_limit),
            "BOOKRECS_COLD_MAX_INTERACTIONS": str(self.cold_max_interactions),
            "BOOKRECS_TRAIN_CANDIDATE_POOL_SIZE": str(self.train_candidate_pool_size),
            "BOOKRECS_TRAIN_PER_SOURCE_LIMIT": str(
                self.train_candidate_per_source_limit
            ),
            "BOOKRECS_TRAIN_PRE_TOP_M": str(self.train_pre_top_m),
            "BOOKRECS_TRAIN_FINAL_TOP_K": str(self.train_final_top_k),
            "BOOKRECS_TRAIN_CF_MAX_NEIGHBORS": str(self.train_cf_max_neighbors),
            "BOOKRECS_TRAIN_CF_MAX_ITEMS_PER_USER": str(
                self.train_cf_max_items_per_user
            ),
            "BOOKRECS_TRAIN_CONTENT_MAX_NEIGHBORS": str(
                self.train_content_max_neighbors
            ),
            "BOOKRECS_TRAIN_PRERANK_MODEL": self.train_prerank_model,
            "BOOKRECS_TRAIN_CATBOOST_ITERATIONS": str(self.train_catboost_iterations),
            "BOOKRECS_TRAIN_CATBOOST_DEPTH": str(self.train_catboost_depth),
            "BOOKRECS_TRAIN_CATBOOST_LEARNING_RATE": str(
                self.train_catboost_learning_rate
            ),
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
    cold_max_interactions: int
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
    train_profile: str
    train_auto_tune: bool
    eval_users_limit: int
    candidate_pool_size: int
    candidate_per_source_limit: int
    pre_top_m: int
    final_top_k: int
    cf_mode: str
    cf_max_neighbors: int
    cf_max_items_per_user: int
    content_max_neighbors: int
    prerank_model: str
    catboost_iterations: int
    catboost_depth: int
    catboost_learning_rate: float
    seed: int

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> PipelineSettings:
        return cls.from_mapping(environ or os.environ)

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> PipelineSettings:
        core = Settings.from_mapping(values)
        auto_tune = core.train_auto_tune
        dataset_name = env_str(values, "BOOKRECS_DATASET_NAME", "goodreads_ya")
        raw_dir = env_str(values, "BOOKRECS_RAW_DIR", "data/raw_data")
        cf_mode = env_str(values, "BOOKRECS_TRAIN_CF_MODE", "auto").lower()
        if cf_mode not in {"auto", "fixed"}:
            raise ValueError("BOOKRECS_TRAIN_CF_MODE must be one of: auto, fixed")
        return cls(
            dataset_name=dataset_name,
            raw_dir=raw_dir,
            books_raw_uri=env_str(
                values, "BOOKRECS_BOOKS_RAW_URI", f"{raw_dir}/books.json.gz"
            ),
            interactions_raw_uri=env_str(
                values,
                "BOOKRECS_INTERACTIONS_RAW_URI",
                f"{raw_dir}/interactions.json.gz",
            ),
            s3_prefix=env_str(
                values, "BOOKRECS_S3_PREFIX", f"s3://bookrecs/datasets/{dataset_name}"
            ),
            k_core=env_positive_int(values, "BOOKRECS_K_CORE", 2),
            keep_recent_fraction=env_float(
                values, "BOOKRECS_KEEP_RECENT_FRACTION", 0.6
            ),
            test_fraction=env_float(values, "BOOKRECS_TEST_FRACTION", 0.25),
            local_val_fraction=env_float(values, "BOOKRECS_LOCAL_VAL_FRACTION", 0.2),
            cold_max_interactions=env_non_negative_int(
                values,
                "BOOKRECS_COLD_MAX_INTERACTIONS",
                core.cold_max_interactions,
            ),
            warm_users_only=env_bool(values, "BOOKRECS_WARM_USERS_ONLY", True),
            language_filter_enabled=env_bool(
                values, "BOOKRECS_LANGUAGE_FILTER_ENABLED", True
            ),
            interactions_chunksize=env_positive_int(
                values, "BOOKRECS_INTERACTIONS_CHUNKSIZE", 200_000
            ),
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
            dataset_dir=env_optional_str(values, "BOOKRECS_TRAIN_DATASET_DIR")
            or f"artifacts/tmp_preprocessed/{dataset_name}",
            output_root=env_str(
                values, "BOOKRECS_TRAIN_OUTPUT_ROOT", core.train_output_root
            ),
            run_name=env_optional_str(values, "BOOKRECS_TRAIN_RUN_NAME"),
            train_profile=env_str(values, "BOOKRECS_TRAIN_PROFILE", core.train_profile),
            train_auto_tune=env_bool(
                values, "BOOKRECS_TRAIN_AUTO_TUNE", core.train_auto_tune
            ),
            eval_users_limit=_read_positive_int(
                values,
                "BOOKRECS_TRAIN_EVAL_USERS_LIMIT",
                core.train_eval_users_limit,
                auto_tune=auto_tune,
            ),
            candidate_pool_size=env_positive_int(
                {} if auto_tune else values,
                "BOOKRECS_TRAIN_CANDIDATE_POOL_SIZE",
                core.train_candidate_pool_size,
            ),
            candidate_per_source_limit=env_positive_int(
                {} if auto_tune else values,
                "BOOKRECS_TRAIN_PER_SOURCE_LIMIT",
                core.train_candidate_per_source_limit,
            ),
            pre_top_m=_read_positive_int(
                values,
                "BOOKRECS_TRAIN_PRE_TOP_M",
                core.train_pre_top_m,
                auto_tune=auto_tune,
            ),
            final_top_k=env_positive_int(
                values, "BOOKRECS_TRAIN_FINAL_TOP_K", core.train_final_top_k
            ),
            cf_mode=cf_mode,
            cf_max_neighbors=_read_positive_int(
                values,
                "BOOKRECS_TRAIN_CF_MAX_NEIGHBORS",
                core.train_cf_max_neighbors,
                auto_tune=auto_tune,
            ),
            cf_max_items_per_user=_read_positive_int(
                values,
                "BOOKRECS_TRAIN_CF_MAX_ITEMS_PER_USER",
                core.train_cf_max_items_per_user,
                auto_tune=auto_tune,
            ),
            content_max_neighbors=_read_positive_int(
                values,
                "BOOKRECS_TRAIN_CONTENT_MAX_NEIGHBORS",
                core.train_content_max_neighbors,
                auto_tune=auto_tune,
            ),
            prerank_model=_read_str(
                values,
                "BOOKRECS_TRAIN_PRERANK_MODEL",
                core.train_prerank_model,
                auto_tune=auto_tune,
            ),
            catboost_iterations=_read_positive_int(
                values,
                "BOOKRECS_TRAIN_CATBOOST_ITERATIONS",
                core.train_catboost_iterations,
                auto_tune=auto_tune,
            ),
            catboost_depth=_read_positive_int(
                values,
                "BOOKRECS_TRAIN_CATBOOST_DEPTH",
                core.train_catboost_depth,
                auto_tune=auto_tune,
            ),
            catboost_learning_rate=_read_positive_float(
                values,
                "BOOKRECS_TRAIN_CATBOOST_LEARNING_RATE",
                core.train_catboost_learning_rate,
                auto_tune=auto_tune,
            ),
            seed=env_positive_int(values, "BOOKRECS_TRAIN_SEED", core.train_seed),
        )


@dataclass(frozen=True)
class ApiRuntimeSettings:
    model_uri: str
    active_model_pointer: str
    auto_reload_sec: int
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
            active_model_pointer=env_str(
                values,
                "BOOKRECS_ACTIVE_MODEL_POINTER",
                "artifacts/runs/active_model.json",
            ),
            auto_reload_sec=env_positive_int(
                values, "BOOKRECS_API_MODEL_AUTO_RELOAD_SEC", 60
            ),
            model_cache_dir=env_str(
                values, "BOOKRECS_API_MODEL_CACHE_DIR", "artifacts/cache/models"
            ),
            s3_region=env_str(values, "BOOKRECS_S3_REGION", "us-east-1"),
            s3_endpoint=env_str(values, "BOOKRECS_S3_ENDPOINT", ""),
            pg_dsn=env_str(values, "BOOKRECS_PG_DSN", ""),
            history_table=env_str(
                values, "BOOKRECS_API_HISTORY_TABLE", "user_item_interactions"
            ),
            inference_log_table=env_str(
                values, "BOOKRECS_API_INFERENCE_LOG_TABLE", "inference_requests"
            ),
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


def env_optional_str(
    values: Mapping[str, str], name: str, default: str | None = None
) -> str | None:
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
        raise ValueError(
            f"Переменная {name} должна быть целым числом, получено: {raw}"
        ) from exc


def env_positive_int(values: Mapping[str, str], name: str, default: int) -> int:
    value = env_int(values, name, default)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def env_non_negative_int(values: Mapping[str, str], name: str, default: int) -> int:
    value = env_int(values, name, default)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _parse_non_negative_int(
    name: str, getter: Callable[[str, str], str], default: int
) -> int:
    value = getter(name, str(default))
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be >= 0, got {value}") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be >= 0, got {parsed}")
    return parsed


def env_positive_float(values: Mapping[str, str], name: str, default: float) -> float:
    value = env_float(values, name, default)
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
        raise ValueError(
            f"Переменная {name} должна быть числом, получено: {raw}"
        ) from exc


def _parse_positive_float(
    name: str, getter: Callable[[str, str], str], default: float
) -> float:
    value = getter(name, str(default))
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be > 0, got {value}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0, got {parsed}")
    return parsed


def _resolve_train_value(
    values: Mapping[str, str],
    name: str,
    default: str,
    *,
    auto_tune: bool,
) -> str:
    if auto_tune:
        return default
    raw = values.get(name)
    if raw is None:
        return default
    value = str(raw).strip()
    return value or default


def _read_positive_int(
    values: Mapping[str, str],
    name: str,
    default: int,
    *,
    auto_tune: bool,
) -> int:
    return env_positive_int({} if auto_tune else values, name, default)


def _read_positive_float(
    values: Mapping[str, str],
    name: str,
    default: float,
    *,
    auto_tune: bool,
) -> float:
    return env_positive_float({} if auto_tune else values, name, default)


def _read_str(
    values: Mapping[str, str],
    name: str,
    default: str,
    *,
    auto_tune: bool,
) -> str:
    return env_str({} if auto_tune else values, name, default)


def _profile_defaults(profile: str) -> dict[str, int | float | str]:
    if profile == "auto":
        detected_profile = _autodetect_resource_profile()
        return _profile_defaults(detected_profile)
    if profile == "lite":
        return {
            "eval_users_limit": 600,
            "candidate_pool_size": 450,
            "candidate_per_source_limit": 120,
            "pre_top_m": 120,
            "cf_max_neighbors": 60,
            "cf_max_items_per_user": 40,
            "content_max_neighbors": 60,
            "prerank_model": "linear",
            "catboost_iterations": 120,
            "catboost_depth": 4,
            "catboost_learning_rate": 0.1,
        }
    return {
        "eval_users_limit": 2000,
        "candidate_pool_size": 1000,
        "candidate_per_source_limit": 300,
        "pre_top_m": 300,
        "cf_max_neighbors": 120,
        "cf_max_items_per_user": 150,
        "content_max_neighbors": 120,
        "prerank_model": "auto",
        "catboost_iterations": 250,
        "catboost_depth": 6,
        "catboost_learning_rate": 0.08,
    }


def _resolve_train_profile(getter: Callable[[str, str], str]) -> str:
    raw = getter("BOOKRECS_TRAIN_PROFILE", "auto").lower()
    if raw in {"default", "lite"}:
        return raw
    if raw != "auto":
        raise ValueError("BOOKRECS_TRAIN_PROFILE must be one of: auto, default, lite")
    return "auto"


def _autodetect_resource_profile() -> str:
    memory_limit_mb = _read_memory_limit_mb()
    if memory_limit_mb is None:
        memory_limit_mb = _read_meminfo_total_mb()
    if memory_limit_mb is None:
        return "lite"
    if memory_limit_mb <= 8 * 1024:
        return "lite"
    return "default"


def _read_memory_limit_mb() -> int | None:
    candidates = [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            raw = open(path, "r", encoding="utf-8").read().strip()
        except OSError:
            continue
        if not raw or raw.lower() == "max":
            continue
        try:
            value_bytes = int(raw)
        except ValueError:
            continue
        if value_bytes <= 0 or value_bytes >= 1 << 60:
            continue
        return value_bytes // (1024 * 1024)
    return None


def _read_meminfo_total_mb() -> int | None:
    path = "/proc/meminfo"
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("MemTotal:"):
                    continue
                value_kb = int(line.split()[1])
                return max(0, value_kb // 1024)
    except (OSError, ValueError, IndexError):
        return None
    return None


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


def _parse_positive_int(
    name: str, get_value: Callable[[str, str], str], default: int
) -> int:
    raw = get_value(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid int for {name}: {raw}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value
