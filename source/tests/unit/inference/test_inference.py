from __future__ import annotations

import pytest

from source.infrastructure.inference.history import UserHistoryProvider
from source.infrastructure.inference.logger import InferenceRequestLogger


@pytest.mark.parametrize(
    "table_name",
    [
        "valid_name",
        "valid_name_2",
        "A_table",
    ],
)
def test_table_name_validation_accepts_safe_identifiers(table_name: str) -> None:
    _ = UserHistoryProvider(pg=None, table_name=table_name)
    _ = InferenceRequestLogger(pg=None, table_name=table_name)


@pytest.mark.parametrize(
    "table_name",
    [
        "bad-name",
        "bad name",
        "123table",
        "a;drop table x",
        "",
    ],
)
def test_table_name_validation_rejects_unsafe_identifiers(table_name: str) -> None:
    with pytest.raises(ValueError):
        _ = UserHistoryProvider(pg=None, table_name=table_name)
    with pytest.raises(ValueError):
        _ = InferenceRequestLogger(pg=None, table_name=table_name)
