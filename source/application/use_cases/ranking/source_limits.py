from __future__ import annotations


def source_limits_for_stage1(history_len: int, per_source_limit: int) -> dict[str, int]:
    base = max(1, int(per_source_limit))
    if history_len <= 1:
        return {
            "cf": max(20, int(base * 0.2)),
            "content": int(base * 1.8),
            "cold": int(base * 1.6),
            "pop": int(base * 1.0),
        }
    if history_len <= 5:
        return {
            "cf": max(40, int(base * 0.65)),
            "content": int(base * 1.4),
            "cold": int(base * 1.15),
            "pop": int(base * 1.0),
        }
    return {
        "cf": base,
        "content": int(base * 1.1),
        "cold": int(base * 0.8),
        "pop": int(base * 0.9),
    }
