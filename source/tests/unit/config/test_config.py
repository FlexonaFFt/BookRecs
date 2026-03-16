from __future__ import annotations

import pytest

from source.infrastructure.config.settings import (
    ApiRuntimeSettings,
    ApiServerSettings,
    EnvSettingsIO,
    PipelineSettings,
    Settings,
    load_settings,
)


def test_load_settings_reads_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    keys = [
        "BOOKRECS_PG_DSN",
        "BOOKRECS_S3_BUCKET",
        "BOOKRECS_S3_REGION",
        "BOOKRECS_S3_ENDPOINT",
        "BOOKRECS_TRAIN_DATASET_DIR",
        "BOOKRECS_TRAIN_OUTPUT_ROOT",
        "BOOKRECS_TRAIN_PROFILE",
        "BOOKRECS_TRAIN_EVAL_USERS_LIMIT",
        "BOOKRECS_COLD_MAX_INTERACTIONS",
        "BOOKRECS_TRAIN_CANDIDATE_POOL_SIZE",
        "BOOKRECS_TRAIN_PER_SOURCE_LIMIT",
        "BOOKRECS_TRAIN_PRE_TOP_M",
        "BOOKRECS_TRAIN_FINAL_TOP_K",
        "BOOKRECS_TRAIN_CF_MAX_NEIGHBORS",
        "BOOKRECS_TRAIN_CF_MAX_ITEMS_PER_USER",
        "BOOKRECS_TRAIN_CONTENT_MAX_NEIGHBORS",
        "BOOKRECS_TRAIN_PRERANK_MODEL",
        "BOOKRECS_TRAIN_CATBOOST_ITERATIONS",
        "BOOKRECS_TRAIN_CATBOOST_DEPTH",
        "BOOKRECS_TRAIN_CATBOOST_LEARNING_RATE",
        "BOOKRECS_TRAIN_SEED",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)

    settings = load_settings()
    assert settings.train_dataset_dir == "artifacts/tmp_preprocessed/goodreads_ya"
    assert settings.train_output_root == "artifacts/runs"
    assert settings.train_profile == "auto"
    assert settings.train_eval_users_limit == 600
    assert settings.cold_max_interactions == 5
    assert settings.train_candidate_pool_size == 450
    assert settings.train_final_top_k == 10
    assert settings.train_prerank_model == "linear"


def test_load_settings_reads_custom_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOOKRECS_PG_DSN", "postgresql://x")
    monkeypatch.setenv("BOOKRECS_S3_BUCKET", "bucket")
    monkeypatch.setenv("BOOKRECS_S3_REGION", "eu-west-1")
    monkeypatch.setenv("BOOKRECS_S3_ENDPOINT", "http://minio:9000")
    monkeypatch.setenv("BOOKRECS_TRAIN_PROFILE", "default")
    monkeypatch.setenv("BOOKRECS_TRAIN_EVAL_USERS_LIMIT", "99")
    monkeypatch.setenv("BOOKRECS_COLD_MAX_INTERACTIONS", "7")
    monkeypatch.setenv("BOOKRECS_TRAIN_CANDIDATE_POOL_SIZE", "777")
    monkeypatch.setenv("BOOKRECS_TRAIN_PER_SOURCE_LIMIT", "111")
    monkeypatch.setenv("BOOKRECS_TRAIN_PRE_TOP_M", "222")
    monkeypatch.setenv("BOOKRECS_TRAIN_FINAL_TOP_K", "33")
    monkeypatch.setenv("BOOKRECS_TRAIN_CF_MAX_NEIGHBORS", "44")
    monkeypatch.setenv("BOOKRECS_TRAIN_CF_MAX_ITEMS_PER_USER", "88")
    monkeypatch.setenv("BOOKRECS_TRAIN_CONTENT_MAX_NEIGHBORS", "55")
    monkeypatch.setenv("BOOKRECS_TRAIN_PRERANK_MODEL", "linear")
    monkeypatch.setenv("BOOKRECS_TRAIN_CATBOOST_ITERATIONS", "123")
    monkeypatch.setenv("BOOKRECS_TRAIN_CATBOOST_DEPTH", "4")
    monkeypatch.setenv("BOOKRECS_TRAIN_CATBOOST_LEARNING_RATE", "0.15")
    monkeypatch.setenv("BOOKRECS_TRAIN_SEED", "66")

    settings = load_settings()
    assert settings.pg_dsn == "postgresql://x"
    assert settings.s3_bucket == "bucket"
    assert settings.s3_region == "eu-west-1"
    assert settings.s3_endpoint == "http://minio:9000"
    assert settings.train_eval_users_limit == 99
    assert settings.cold_max_interactions == 7
    assert settings.train_candidate_pool_size == 777
    assert settings.train_candidate_per_source_limit == 111
    assert settings.train_pre_top_m == 222
    assert settings.train_final_top_k == 33
    assert settings.train_cf_max_neighbors == 44
    assert settings.train_cf_max_items_per_user == 88
    assert settings.train_content_max_neighbors == 55
    assert settings.train_prerank_model == "linear"
    assert settings.train_catboost_iterations == 123
    assert settings.train_catboost_depth == 4
    assert settings.train_catboost_learning_rate == pytest.approx(0.15)
    assert settings.train_seed == 66


@pytest.mark.parametrize(
    "key,value",
    [
        ("BOOKRECS_TRAIN_EVAL_USERS_LIMIT", "0"),
        ("BOOKRECS_COLD_MAX_INTERACTIONS", "-1"),
        ("BOOKRECS_TRAIN_PROFILE", "bad"),
        ("BOOKRECS_TRAIN_CANDIDATE_POOL_SIZE", "-1"),
        ("BOOKRECS_TRAIN_PER_SOURCE_LIMIT", "abc"),
    ],
)
def test_load_settings_raises_for_invalid_positive_ints(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
) -> None:
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        load_settings()


def test_env_settings_io_read_and_write() -> None:
    env = {
        "BOOKRECS_PG_DSN": "postgresql://local",
        "BOOKRECS_S3_BUCKET": "bookrecs",
        "BOOKRECS_S3_REGION": "us-east-1",
        "BOOKRECS_S3_ENDPOINT": "http://localhost:9000",
        "BOOKRECS_TRAIN_DATASET_DIR": "artifacts/tmp_preprocessed/goodreads_ya",
        "BOOKRECS_TRAIN_OUTPUT_ROOT": "artifacts/runs",
        "BOOKRECS_TRAIN_PROFILE": "default",
        "BOOKRECS_TRAIN_EVAL_USERS_LIMIT": "123",
        "BOOKRECS_COLD_MAX_INTERACTIONS": "5",
        "BOOKRECS_TRAIN_CANDIDATE_POOL_SIZE": "1000",
        "BOOKRECS_TRAIN_PER_SOURCE_LIMIT": "300",
        "BOOKRECS_TRAIN_PRE_TOP_M": "300",
        "BOOKRECS_TRAIN_FINAL_TOP_K": "10",
        "BOOKRECS_TRAIN_CF_MAX_NEIGHBORS": "120",
        "BOOKRECS_TRAIN_CF_MAX_ITEMS_PER_USER": "150",
        "BOOKRECS_TRAIN_CONTENT_MAX_NEIGHBORS": "120",
        "BOOKRECS_TRAIN_PRERANK_MODEL": "auto",
        "BOOKRECS_TRAIN_CATBOOST_ITERATIONS": "250",
        "BOOKRECS_TRAIN_CATBOOST_DEPTH": "6",
        "BOOKRECS_TRAIN_CATBOOST_LEARNING_RATE": "0.08",
        "BOOKRECS_TRAIN_SEED": "42",
    }
    io = EnvSettingsIO(environ=env)
    settings = io.read()
    assert settings.train_eval_users_limit == 123

    updated = Settings(
        pg_dsn=settings.pg_dsn,
        s3_bucket=settings.s3_bucket,
        s3_region=settings.s3_region,
        s3_endpoint=settings.s3_endpoint,
        train_dataset_dir=settings.train_dataset_dir,
        train_output_root=settings.train_output_root,
        train_profile=settings.train_profile,
        train_auto_tune=settings.train_auto_tune,
        train_eval_users_limit=777,
        cold_max_interactions=settings.cold_max_interactions,
        train_candidate_pool_size=settings.train_candidate_pool_size,
        train_candidate_per_source_limit=settings.train_candidate_per_source_limit,
        train_pre_top_m=settings.train_pre_top_m,
        train_final_top_k=settings.train_final_top_k,
        train_cf_max_neighbors=settings.train_cf_max_neighbors,
        train_cf_max_items_per_user=settings.train_cf_max_items_per_user,
        train_content_max_neighbors=settings.train_content_max_neighbors,
        train_prerank_model=settings.train_prerank_model,
        train_catboost_iterations=settings.train_catboost_iterations,
        train_catboost_depth=settings.train_catboost_depth,
        train_catboost_learning_rate=settings.train_catboost_learning_rate,
        train_seed=settings.train_seed,
    )
    io.write(updated)
    assert env["BOOKRECS_TRAIN_EVAL_USERS_LIMIT"] == "777"
    assert env["BOOKRECS_COLD_MAX_INTERACTIONS"] == "5"


def test_api_runtime_settings_reads_defaults() -> None:
    settings = ApiRuntimeSettings.from_mapping({})
    assert settings.model_uri == ""
    assert settings.model_cache_dir == "artifacts/cache/models"
    assert settings.s3_region == "us-east-1"
    assert settings.history_table == "user_item_interactions"


def test_api_server_settings_raises_for_invalid_port() -> None:
    with pytest.raises(ValueError):
        ApiServerSettings.from_mapping({"BOOKRECS_API_PORT": "0"})


def test_pipeline_settings_validates_cf_mode() -> None:
    with pytest.raises(ValueError):
        PipelineSettings.from_mapping({"BOOKRECS_TRAIN_CF_MODE": "bad"})


def test_pipeline_settings_reads_cold_threshold() -> None:
    settings = PipelineSettings.from_mapping({"BOOKRECS_COLD_MAX_INTERACTIONS": "9"})
    assert settings.cold_max_interactions == 9


def test_load_settings_lite_profile_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOOKRECS_TRAIN_PROFILE", "lite")
    settings = load_settings()
    assert settings.train_profile == "lite"
    assert settings.train_candidate_pool_size == 450
    assert settings.train_candidate_per_source_limit == 120
    assert settings.train_pre_top_m == 120
    assert settings.train_cf_max_neighbors == 60
    assert settings.train_cf_max_items_per_user == 40
    assert settings.train_content_max_neighbors == 60
    assert settings.train_prerank_model == "linear"


def test_load_settings_auto_profile_without_cgroup_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOOKRECS_TRAIN_PROFILE", raising=False)
    monkeypatch.setattr("source.infrastructure.config.settings._read_memory_limit_mb", lambda: None)
    monkeypatch.setattr("source.infrastructure.config.settings._read_meminfo_total_mb", lambda: None)
    settings = load_settings()
    assert settings.train_profile == "auto"
    assert settings.train_candidate_pool_size == 450
    assert settings.train_prerank_model == "linear"


def test_load_settings_auto_profile_with_small_memory_uses_lite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOOKRECS_TRAIN_PROFILE", "auto")
    monkeypatch.setattr("source.infrastructure.config.settings._read_memory_limit_mb", lambda: 4096)
    settings = load_settings()
    assert settings.train_profile == "auto"
    assert settings.train_candidate_pool_size == 450
    assert settings.train_cf_max_items_per_user == 40
    assert settings.train_prerank_model == "linear"


def test_load_settings_auto_profile_uses_meminfo_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOOKRECS_TRAIN_PROFILE", "auto")
    monkeypatch.setattr("source.infrastructure.config.settings._read_memory_limit_mb", lambda: None)
    monkeypatch.setattr("source.infrastructure.config.settings._read_meminfo_total_mb", lambda: 6144)
    settings = load_settings()
    assert settings.train_candidate_pool_size == 450
    assert settings.train_prerank_model == "linear"
