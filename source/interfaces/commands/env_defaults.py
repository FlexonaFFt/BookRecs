from __future__ import annotations

import os


TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}


# Читает значение окружения как строку, если оно задано.
def env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


# Читает значение окружения и возвращает None для пустых значений.
def env_optional_str(name: str, default: str | None = None) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value or default


# Читает целое число из окружения.
def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Переменная {name} должна быть целым числом, получено: {raw}") from exc


# Читает число с плавающей точкой из окружения.
def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Переменная {name} должна быть числом, получено: {raw}") from exc


# Читает булево значение из окружения.
def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if not value:
        return default
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    raise ValueError(f"Переменная {name} должна быть булевой, получено: {raw}")
