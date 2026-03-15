from __future__ import annotations


def source_limits_for_stage1(history_len: int, per_source_limit: int) -> dict[str, int]:
    base = max(1, int(per_source_limit))
    if history_len <= 1:
        return {
            "cf": max(20, int(base * 0.2)),
            "content": int(base * 2.4),
            "cold": int(base * 2.2),
            "pop": int(base * 0.7),
        }
    if history_len <= 5:
        return {
            "cf": max(40, int(base * 0.55)),
            "content": int(base * 1.8),
            "cold": int(base * 1.5),
            "pop": int(base * 0.8),
        }
    return {
        "cf": base,
        "content": int(base * 1.3),
        "cold": base,
        "pop": int(base * 0.7),
    }


def source_min_quota_for_stage1(history_len: int, pool_size: int) -> dict[str, int]:
    pool = max(1, int(pool_size))
    if history_len <= 1:
        return {
            "content": max(1, int(pool * 0.35)),
            "cold": max(1, int(pool * 0.20)),
        }
    if history_len <= 5:
        return {
            "content": max(1, int(pool * 0.25)),
            "cold": max(1, int(pool * 0.15)),
        }
    return {
        "content": max(1, int(pool * 0.18)),
        "cold": max(1, int(pool * 0.10)),
    }
