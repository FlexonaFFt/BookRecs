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


def _parse_positive_int(name: str, get_value: Callable[[str, str], str], default: int) -> int:
    raw = get_value(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid int for {name}: {raw}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value
