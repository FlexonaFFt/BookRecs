from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

from source.infrastructure.storage.postgres.postgres_client import PostgresClient


@dataclass(frozen=True)
class OfflineExperiment:
    experiment_id: str
    description: str = ""
    holdout_version_id: str = ""
    created_at: datetime | None = None


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    model_tag: str
    role: str = "test"
    split: str = "overall"
    k: int = 10
    ndcg_at_k: float | None = None
    recall_at_k: float | None = None
    coverage_at_k: float | None = None
    extra_json: dict = field(default_factory=dict)


# Сохраняет результаты offline-экспериментов (эмуляция пилота) в PostgreSQL.
class PostgresExperimentLog:

    def __init__(self, pg: PostgresClient) -> None:
        self._pg = pg

    def create_experiment(self, experiment: OfflineExperiment) -> None:
        self._pg.execute(
            """
            INSERT INTO offline_experiments (
                experiment_id, description, holdout_version_id, created_at
            )
            VALUES (%s, %s, %s, COALESCE(%s, NOW()))
            ON CONFLICT (experiment_id) DO UPDATE SET
                description = EXCLUDED.description,
                holdout_version_id = EXCLUDED.holdout_version_id
            """,
            (
                experiment.experiment_id,
                experiment.description,
                experiment.holdout_version_id,
                experiment.created_at,
            ),
        )

    def upsert_result(self, result: ExperimentResult) -> None:
        self._pg.execute(
            """
            INSERT INTO experiment_results (
                experiment_id, model_tag, role, split, k,
                ndcg_at_k, recall_at_k, coverage_at_k, extra_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (experiment_id, model_tag, split) DO UPDATE SET
                role = EXCLUDED.role,
                k = EXCLUDED.k,
                ndcg_at_k = EXCLUDED.ndcg_at_k,
                recall_at_k = EXCLUDED.recall_at_k,
                coverage_at_k = EXCLUDED.coverage_at_k,
                extra_json = EXCLUDED.extra_json
            """,
            (
                result.experiment_id,
                result.model_tag,
                result.role,
                result.split,
                result.k,
                result.ndcg_at_k,
                result.recall_at_k,
                result.coverage_at_k,
                json.dumps(result.extra_json, ensure_ascii=False),
            ),
        )

    def get_experiment(self, experiment_id: str) -> OfflineExperiment | None:
        row = self._pg.fetchone(
            """
            SELECT experiment_id, description, holdout_version_id, created_at
            FROM offline_experiments
            WHERE experiment_id = %s
            """,
            (experiment_id,),
        )
        if row is None:
            return None
        return OfflineExperiment(
            experiment_id=str(row["experiment_id"]),
            description=str(row.get("description", "") or ""),
            holdout_version_id=str(row.get("holdout_version_id", "") or ""),
            created_at=_as_datetime_or_none(row.get("created_at")),
        )

    def list_results(self, experiment_id: str) -> list[ExperimentResult]:
        rows = self._pg.fetchall(
            """
            SELECT experiment_id, model_tag, role, split, k,
                   ndcg_at_k, recall_at_k, coverage_at_k, extra_json
            FROM experiment_results
            WHERE experiment_id = %s
            ORDER BY model_tag, split
            """,
            (experiment_id,),
        )
        return [_row_to_result(row) for row in rows]


def _row_to_result(row: dict) -> ExperimentResult:
    extra = row.get("extra_json") or {}
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except Exception:
            extra = {}
    return ExperimentResult(
        experiment_id=str(row["experiment_id"]),
        model_tag=str(row["model_tag"]),
        role=str(row.get("role", "test")),
        split=str(row.get("split", "overall")),
        k=int(row.get("k", 10)),
        ndcg_at_k=_as_float_or_none(row.get("ndcg_at_k")),
        recall_at_k=_as_float_or_none(row.get("recall_at_k")),
        coverage_at_k=_as_float_or_none(row.get("coverage_at_k")),
        extra_json=extra if isinstance(extra, dict) else {},
    )


def _as_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_datetime_or_none(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None
