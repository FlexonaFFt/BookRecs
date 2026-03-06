from __future__ import annotations

import pytest

from source.infrastructure.config.settings import load_settings


def test_load_settings_reads_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    keys = [
        "BOOKRECS_PG_DSN",
        "BOOKRECS_S3_BUCKET",
        "BOOKRECS_S3_REGION",
        "BOOKRECS_S3_ENDPOINT",
        "BOOKRECS_TRAIN_DATASET_DIR",
        "BOOKRECS_TRAIN_OUTPUT_ROOT",
        "BOOKRECS_TRAIN_EVAL_USERS_LIMIT",
        "BOOKRECS_TRAIN_CANDIDATE_POOL_SIZE",
        "BOOKRECS_TRAIN_PER_SOURCE_LIMIT",
        "BOOKRECS_TRAIN_PRE_TOP_M",
        "BOOKRECS_TRAIN_FINAL_TOP_K",
        "BOOKRECS_TRAIN_CF_MAX_NEIGHBORS",
        "BOOKRECS_TRAIN_CONTENT_MAX_NEIGHBORS",
        "BOOKRECS_TRAIN_SEED",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)

    settings = load_settings()
    assert settings.train_dataset_dir == "artifacts/tmp_preprocessed/goodreads_ya"
    assert settings.train_output_root == "artifacts/runs"
    assert settings.train_eval_users_limit == 2000
    assert settings.train_candidate_pool_size == 1000
    assert settings.train_final_top_k == 10


def test_load_settings_reads_custom_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOOKRECS_PG_DSN", "postgresql://x")
    monkeypatch.setenv("BOOKRECS_S3_BUCKET", "bucket")
    monkeypatch.setenv("BOOKRECS_S3_REGION", "eu-west-1")
    monkeypatch.setenv("BOOKRECS_S3_ENDPOINT", "http://minio:9000")
    monkeypatch.setenv("BOOKRECS_TRAIN_EVAL_USERS_LIMIT", "99")
    monkeypatch.setenv("BOOKRECS_TRAIN_CANDIDATE_POOL_SIZE", "777")
    monkeypatch.setenv("BOOKRECS_TRAIN_PER_SOURCE_LIMIT", "111")
    monkeypatch.setenv("BOOKRECS_TRAIN_PRE_TOP_M", "222")
    monkeypatch.setenv("BOOKRECS_TRAIN_FINAL_TOP_K", "33")
    monkeypatch.setenv("BOOKRECS_TRAIN_CF_MAX_NEIGHBORS", "44")
    monkeypatch.setenv("BOOKRECS_TRAIN_CONTENT_MAX_NEIGHBORS", "55")
    monkeypatch.setenv("BOOKRECS_TRAIN_SEED", "66")

    settings = load_settings()
    assert settings.pg_dsn == "postgresql://x"
    assert settings.s3_bucket == "bucket"
    assert settings.s3_region == "eu-west-1"
    assert settings.s3_endpoint == "http://minio:9000"
    assert settings.train_eval_users_limit == 99
    assert settings.train_candidate_pool_size == 777
    assert settings.train_candidate_per_source_limit == 111
    assert settings.train_pre_top_m == 222
    assert settings.train_final_top_k == 33
    assert settings.train_cf_max_neighbors == 44
    assert settings.train_content_max_neighbors == 55
    assert settings.train_seed == 66


@pytest.mark.parametrize(
    "key,value",
    [
        ("BOOKRECS_TRAIN_EVAL_USERS_LIMIT", "0"),
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
