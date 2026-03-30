from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from source.infrastructure.storage.experiments.postgres_experiment_log import (
    ExperimentResult,
    OfflineExperiment,
    PostgresExperimentLog,
    _as_float_or_none,
    _row_to_result,
)


class FakePostgresClient:
    def __init__(self) -> None:
        self.executions: list[tuple[str, tuple]] = []
        self._fetchone_result: dict[str, Any] | None = None
        self._fetchall_result: list[dict[str, Any]] = []

    def execute(self, query: str, params: tuple = ()) -> None:
        self.executions.append((query.strip(), params))

    def fetchone(self, query: str, params: tuple = ()) -> dict[str, Any] | None:
        self.executions.append((query.strip(), params))
        return self._fetchone_result

    def fetchall(self, query: str, params: tuple = ()) -> list[dict[str, Any]]:
        self.executions.append((query.strip(), params))
        return self._fetchall_result

    def executed_queries(self) -> list[str]:
        return [q for q, _ in self.executions]

    def all_params(self) -> list[tuple]:
        return [p for _, p in self.executions]


# --- create_experiment ---


def test_create_experiment_executes_insert() -> None:
    fake = FakePostgresClient()
    log = PostgresExperimentLog(fake)
    exp = OfflineExperiment(
        experiment_id="exp-001",
        description="test run",
        holdout_version_id="v-abc123",
    )
    log.create_experiment(exp)

    queries = fake.executed_queries()
    assert any("INSERT INTO offline_experiments" in q for q in queries)
    params = fake.all_params()[0]
    assert params[0] == "exp-001"
    assert params[1] == "test run"
    assert params[2] == "v-abc123"


def test_create_experiment_upserts_on_conflict() -> None:
    fake = FakePostgresClient()
    log = PostgresExperimentLog(fake)
    exp = OfflineExperiment(experiment_id="exp-001")
    log.create_experiment(exp)
    queries = fake.executed_queries()
    assert any("ON CONFLICT" in q for q in queries)


# --- upsert_result ---


def test_upsert_result_executes_insert() -> None:
    fake = FakePostgresClient()
    log = PostgresExperimentLog(fake)
    result = ExperimentResult(
        experiment_id="exp-001",
        model_tag="baseline_popular",
        role="control",
        split="overall",
        k=10,
        ndcg_at_k=0.12,
        recall_at_k=0.25,
        coverage_at_k=0.80,
    )
    log.upsert_result(result)

    queries = fake.executed_queries()
    assert any("INSERT INTO experiment_results" in q for q in queries)
    params = fake.all_params()[0]
    assert params[0] == "exp-001"
    assert params[1] == "baseline_popular"
    assert params[2] == "control"
    assert params[3] == "overall"
    assert params[5] == pytest.approx(0.12)
    assert params[6] == pytest.approx(0.25)


def test_upsert_result_upserts_on_conflict() -> None:
    fake = FakePostgresClient()
    log = PostgresExperimentLog(fake)
    result = ExperimentResult(experiment_id="exp-001", model_tag="m1")
    log.upsert_result(result)
    queries = fake.executed_queries()
    assert any("ON CONFLICT" in q and "DO UPDATE" in q for q in queries)


# --- get_experiment ---


def test_get_experiment_returns_none_when_not_found() -> None:
    fake = FakePostgresClient()
    fake._fetchone_result = None
    log = PostgresExperimentLog(fake)
    assert log.get_experiment("exp-999") is None


def test_get_experiment_returns_experiment_when_found() -> None:
    ts = datetime(2025, 3, 30, 12, 0, 0, tzinfo=timezone.utc)
    fake = FakePostgresClient()
    fake._fetchone_result = {
        "experiment_id": "exp-001",
        "description": "hello",
        "holdout_version_id": "v1",
        "created_at": ts,
    }
    log = PostgresExperimentLog(fake)
    exp = log.get_experiment("exp-001")

    assert exp is not None
    assert exp.experiment_id == "exp-001"
    assert exp.description == "hello"
    assert exp.holdout_version_id == "v1"
    assert exp.created_at == ts


# --- list_results ---


def test_list_results_returns_empty_when_no_rows() -> None:
    fake = FakePostgresClient()
    fake._fetchall_result = []
    log = PostgresExperimentLog(fake)
    assert log.list_results("exp-001") == []


def test_list_results_maps_rows_correctly() -> None:
    fake = FakePostgresClient()
    fake._fetchall_result = [
        {
            "experiment_id": "exp-001",
            "model_tag": "hybrid_3stage",
            "role": "test",
            "split": "cold",
            "k": 10,
            "ndcg_at_k": 0.09,
            "recall_at_k": 0.18,
            "coverage_at_k": 0.55,
            "extra_json": {},
        }
    ]
    log = PostgresExperimentLog(fake)
    results = log.list_results("exp-001")

    assert len(results) == 1
    r = results[0]
    assert r.model_tag == "hybrid_3stage"
    assert r.split == "cold"
    assert r.ndcg_at_k == pytest.approx(0.09)
    assert r.recall_at_k == pytest.approx(0.18)


def test_list_results_parses_extra_json_string() -> None:
    import json

    fake = FakePostgresClient()
    fake._fetchall_result = [
        {
            "experiment_id": "exp-001",
            "model_tag": "m1",
            "role": "test",
            "split": "overall",
            "k": 10,
            "ndcg_at_k": None,
            "recall_at_k": None,
            "coverage_at_k": None,
            "extra_json": json.dumps({"custom": 1.0}),
        }
    ]
    log = PostgresExperimentLog(fake)
    results = log.list_results("exp-001")
    assert results[0].extra_json == {"custom": 1.0}


# --- helpers ---


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.5, 0.5),
        ("0.75", 0.75),
        (None, None),
        ("bad", None),
    ],
)
def test_as_float_or_none(value: Any, expected: float | None) -> None:
    result = _as_float_or_none(value)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_row_to_result_handles_none_metrics() -> None:
    row = {
        "experiment_id": "e1",
        "model_tag": "m1",
        "role": "test",
        "split": "overall",
        "k": 10,
        "ndcg_at_k": None,
        "recall_at_k": None,
        "coverage_at_k": None,
        "extra_json": None,
    }
    result = _row_to_result(row)
    assert result.ndcg_at_k is None
    assert result.extra_json == {}
