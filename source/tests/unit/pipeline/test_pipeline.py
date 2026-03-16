from __future__ import annotations

import os

import pytest

from source.infrastructure.config import env_bool, env_float, env_int, env_optional_str, env_str
from source.infrastructure.config import env_non_negative_int, env_positive_float


def test_env_str_returns_default_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("X_NAME", raising=False)
    assert env_str(os.environ, "X_NAME", "default") == "default"


def test_env_optional_str_handles_blank_and_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("X_OPT", raising=False)
    assert env_optional_str(os.environ, "X_OPT", "fallback") == "fallback"

    monkeypatch.setenv("X_OPT", "   ")
    assert env_optional_str(os.environ, "X_OPT", "fallback") == "fallback"

    monkeypatch.setenv("X_OPT", " value ")
    assert env_optional_str(os.environ, "X_OPT", "fallback") == "value"


def test_env_int_parses_or_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("X_INT", raising=False)
    assert env_int(os.environ, "X_INT", 5) == 5

    monkeypatch.setenv("X_INT", " 42 ")
    assert env_int(os.environ, "X_INT", 5) == 42

    monkeypatch.setenv("X_INT", "")
    assert env_int(os.environ, "X_INT", 5) == 5


def test_env_int_raises_for_invalid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X_INT", "nan")
    with pytest.raises(ValueError):
        env_int(os.environ, "X_INT", 1)


def test_env_float_parses_or_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("X_FLOAT", raising=False)
    assert env_float(os.environ, "X_FLOAT", 0.1) == pytest.approx(0.1)

    monkeypatch.setenv("X_FLOAT", " 0.75 ")
    assert env_float(os.environ, "X_FLOAT", 0.1) == pytest.approx(0.75)

    monkeypatch.setenv("X_FLOAT", "")
    assert env_float(os.environ, "X_FLOAT", 0.1) == pytest.approx(0.1)


def test_env_float_raises_for_invalid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X_FLOAT", "oops")
    with pytest.raises(ValueError):
        env_float(os.environ, "X_FLOAT", 0.1)


def test_env_non_negative_int_accepts_zero_and_rejects_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X_NN_INT", "0")
    assert env_non_negative_int(os.environ, "X_NN_INT", 5) == 0

    monkeypatch.setenv("X_NN_INT", "-1")
    with pytest.raises(ValueError):
        env_non_negative_int(os.environ, "X_NN_INT", 5)


def test_env_positive_float_accepts_positive_and_rejects_non_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X_POS_FLOAT", "0.2")
    assert env_positive_float(os.environ, "X_POS_FLOAT", 0.1) == pytest.approx(0.2)

    monkeypatch.setenv("X_POS_FLOAT", "0")
    with pytest.raises(ValueError):
        env_positive_float(os.environ, "X_POS_FLOAT", 0.1)


@pytest.mark.parametrize("value", ["1", "true", "yes", "y", "on", "TRUE"])
def test_env_bool_true_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("X_BOOL", value)
    assert env_bool(os.environ, "X_BOOL", False) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "n", "off", "FALSE"])
def test_env_bool_false_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("X_BOOL", value)
    assert env_bool(os.environ, "X_BOOL", True) is False


def test_env_bool_fallback_and_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("X_BOOL", raising=False)
    assert env_bool(os.environ, "X_BOOL", True) is True

    monkeypatch.setenv("X_BOOL", "")
    assert env_bool(os.environ, "X_BOOL", False) is False

    monkeypatch.setenv("X_BOOL", "sometimes")
    with pytest.raises(ValueError):
        env_bool(os.environ, "X_BOOL", False)
