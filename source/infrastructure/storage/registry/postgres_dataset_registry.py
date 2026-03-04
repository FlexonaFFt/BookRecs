from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime

from source.application.ports import DatasetRegistryPort
from source.domain.entities import DatasetVersion, PreprocessingParams
from source.infrastructure.storage.postgres.postgres_client import PostgresClient
# Сохраняет версии датасета в PostgreSQL.
class PostgresDatasetRegistry(DatasetRegistryPort):


    def __init__(self, pg: PostgresClient) -> None:
        self._pg = pg

    def find_success_by_hash(self, dataset_name: str, params_hash: str) -> DatasetVersion | None:
        row = self._pg.fetchone(
            """
            SELECT dataset_name, version_id, params_hash, s3_prefix, params_json, stats_json, metadata_json, created_at
            FROM dataset_registry
            WHERE dataset_name = %s AND params_hash = %s
            """,
            (dataset_name, params_hash),
        )
        if row is None:
            return None

        params_dict = _loads_json(row["params_json"])
        stats_dict = _loads_json(row["stats_json"])
        metadata_dict = _loads_json(row["metadata_json"])

        return DatasetVersion(
            dataset_name=row["dataset_name"],
            version_id=row["version_id"],
            params_hash=row["params_hash"],
            s3_prefix=row["s3_prefix"],
            params=PreprocessingParams(**params_dict),
            created_at=_as_datetime(row["created_at"]),
            stats={str(k): int(v) for k, v in stats_dict.items()},
            metadata=metadata_dict,
        )

    def upsert(self, dataset_version: DatasetVersion) -> None:
        self._pg.execute(
            """
            INSERT INTO dataset_registry (
                dataset_name, params_hash, version_id, s3_prefix,
                params_json, stats_json, metadata_json, created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, NOW())
            ON CONFLICT (dataset_name, params_hash) DO UPDATE SET
                version_id = EXCLUDED.version_id,
                s3_prefix = EXCLUDED.s3_prefix,
                params_json = EXCLUDED.params_json,
                stats_json = EXCLUDED.stats_json,
                metadata_json = EXCLUDED.metadata_json,
                created_at = EXCLUDED.created_at,
                updated_at = NOW()
            """,
            (
                dataset_version.dataset_name,
                dataset_version.params_hash,
                dataset_version.version_id,
                dataset_version.s3_prefix,
                json.dumps(asdict(dataset_version.params), ensure_ascii=False),
                json.dumps(dataset_version.stats, ensure_ascii=False),
                json.dumps(dataset_version.metadata, ensure_ascii=False),
                dataset_version.created_at,
            ),
        )


def _loads_json(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return {}


def _as_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise ValueError(f"Unexpected datetime value: {value!r}")
