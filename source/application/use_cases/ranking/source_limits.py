from __future__ import annotations


def source_limits_for_stage1(history_len: int, per_source_limit: int) -> dict[str, int]:
    base = max(1, int(per_source_limit))
    if history_len <= 1:
        return {
            "cf": max(15, int(base * 0.15)),
            "content": int(base * 2.8),
            "cold": int(base * 2.6),
            "pop": int(base * 0.6),
        }
    if history_len <= 5:
        return {
            "cf": max(30, int(base * 0.45)),
            "content": int(base * 2.1),
            "cold": int(base * 1.9),
            "pop": int(base * 0.7),
        }
    return {
        "cf": int(base * 0.85),
        "content": int(base * 1.6),
        "cold": int(base * 1.5),
        "pop": int(base * 0.6),
    }


def source_min_quota_for_stage1(history_len: int, pool_size: int) -> dict[str, int]:
    pool = max(1, int(pool_size))
    if history_len <= 1:
        return {
            "content": max(1, int(pool * 0.42)),
            "cold": max(1, int(pool * 0.28)),
        }
    if history_len <= 5:
        return {
            "content": max(1, int(pool * 0.32)),
            "cold": max(1, int(pool * 0.22)),
        }
    return {
        "content": max(1, int(pool * 0.24)),
        "cold": max(1, int(pool * 0.18)),
    }
