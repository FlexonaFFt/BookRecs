from __future__ import annotations

import pytest

from source.interfaces.pipeline_entrypoint import _env_bool, _env_float, _env_int, _env_optional_str, _env_str


def test_env_str_returns_default_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("X_NAME", raising=False)
    assert _env_str("X_NAME", "default") == "default"


def test_env_optional_str_handles_blank_and_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("X_OPT", raising=False)
    assert _env_optional_str("X_OPT", "fallback") == "fallback"

    monkeypatch.setenv("X_OPT", "   ")
    assert _env_optional_str("X_OPT", "fallback") == "fallback"

    monkeypatch.setenv("X_OPT", " value ")
    assert _env_optional_str("X_OPT", "fallback") == "value"


def test_env_int_parses_or_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("X_INT", raising=False)
    assert _env_int("X_INT", 5) == 5

    monkeypatch.setenv("X_INT", " 42 ")
    assert _env_int("X_INT", 5) == 42

    monkeypatch.setenv("X_INT", "")
    assert _env_int("X_INT", 5) == 5


def test_env_int_raises_for_invalid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X_INT", "nan")
    with pytest.raises(ValueError):
        _env_int("X_INT", 1)


def test_env_float_parses_or_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("X_FLOAT", raising=False)
    assert _env_float("X_FLOAT", 0.1) == pytest.approx(0.1)

    monkeypatch.setenv("X_FLOAT", " 0.75 ")
    assert _env_float("X_FLOAT", 0.1) == pytest.approx(0.75)

    monkeypatch.setenv("X_FLOAT", "")
    assert _env_float("X_FLOAT", 0.1) == pytest.approx(0.1)


def test_env_float_raises_for_invalid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X_FLOAT", "oops")
    with pytest.raises(ValueError):
        _env_float("X_FLOAT", 0.1)


@pytest.mark.parametrize("value", ["1", "true", "yes", "y", "on", "TRUE"])
def test_env_bool_true_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("X_BOOL", value)
    assert _env_bool("X_BOOL", False) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "n", "off", "FALSE"])
def test_env_bool_false_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("X_BOOL", value)
    assert _env_bool("X_BOOL", True) is False


def test_env_bool_fallback_and_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("X_BOOL", raising=False)
    assert _env_bool("X_BOOL", True) is True

    monkeypatch.setenv("X_BOOL", "")
    assert _env_bool("X_BOOL", False) is False

    monkeypatch.setenv("X_BOOL", "sometimes")
    with pytest.raises(ValueError):
        _env_bool("X_BOOL", False)
