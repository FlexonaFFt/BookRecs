from __future__ import annotations

import itertools
import math
import os
from collections import Counter, defaultdict
from typing import Any

from source.application.use_cases.training.common.data_ops import build_item_interaction_counts


def fit_stage1(data: dict[str, Any], cmd: Any, logger: Any) -> dict[str, Any]:
    train = data["local_train"]
    books = data["books"]
    logger.start_step("stage1_fit", total=4)

    pop_items, pop_scores = fit_popularity(train)
    logger.progress("stage1_fit", done=1, total=4)

    cf_neighbors = fit_cf_neighbors(
        interactions=train,
        cf_mode=getattr(cmd, "cf_mode", "auto"),
        max_neighbors=cmd.cf_max_neighbors,
        max_items_per_user=getattr(cmd, "cf_max_items_per_user", 150),
        batch_size=getattr(cmd, "cf_batch_size", 2000),
        retained_neighbors_factor=getattr(cmd, "cf_retained_neighbors_factor", 4),
        logger=logger,
    )
    logger.progress("stage1_fit", done=2, total=4)

    content_similar = fit_content_neighbors(
        books=books,
        max_neighbors=cmd.content_max_neighbors,
        logger=logger,
    )
    logger.progress("stage1_fit", done=3, total=4)

    metadata_retrieval = fit_metadata_retrieval(
        books=books,
        logger=logger,
    )
    logger.progress("stage1_fit", done=4, total=4)
    item_interaction_counts = build_item_interaction_counts(train)
    catalog_item_ids = set(books["item_id"].dropna().tolist())
    cold_item_ids = {
        item_id
        for item_id in catalog_item_ids
        if int(item_interaction_counts.get(item_id, 0)) <= int(getattr(cmd, "cold_max_interactions", 0))
    }
    logger.end_step(
        "stage1_fit",
        status="SUCCESS",
        pop_items=len(pop_items),
        cf_items=len(cf_neighbors),
        content_items=len(content_similar),
        metadata_items=len(metadata_retrieval["item_metadata"]),
        cold_items=len(cold_item_ids),
    )
    return {
        "pop_items": pop_items,
        "pop_scores": pop_scores,
        "cf_neighbors": cf_neighbors,
        "content_similar": content_similar,
        "item_interaction_counts": item_interaction_counts,
        "cold_item_ids": cold_item_ids,
        "cold_max_interactions": int(getattr(cmd, "cold_max_interactions", 0)),
        "item_metadata": metadata_retrieval["item_metadata"],
        "author_index": metadata_retrieval["author_index"],
        "series_index": metadata_retrieval["series_index"],
        "tag_index": metadata_retrieval["tag_index"],
    }


def fit_popularity(train: Any) -> tuple[list[Any], dict[Any, float]]:
    pop = train.groupby("item_id", as_index=False).size().rename(columns={"size": "n"})
    pop = pop.sort_values("n", ascending=False).reset_index(drop=True)
    top_items = pop["item_id"].tolist()
    max_n = float(pop["n"].max()) if len(pop) else 1.0
    if max_n <= 0:
        max_n = 1.0
    scores = {row.item_id: float(row.n / max_n) for row in pop.itertuples(index=False)}
    return top_items, scores


def fit_cf_neighbors(
    interactions: Any,
    cf_mode: str,
    max_neighbors: int,
    max_items_per_user: int,
    batch_size: int,
    retained_neighbors_factor: int,
    logger: Any,
) -> dict[Any, list[tuple[Any, float]]]:
    user_items = (
        interactions[["user_id", "item_id"]]
        .drop_duplicates(["user_id", "item_id"])
        .groupby("user_id", sort=False)["item_id"]
        .agg(list)
    )
    neighbor_counts: dict[Any, Counter[Any]] = defaultdict(Counter)
    item_counts: Counter[Any] = Counter()
    truncated_users = 0
    effective_mode = cf_mode if cf_mode in {"auto", "fixed"} else "auto"
    memory_limit_mb = _read_memory_limit_mb()
    current_cap = _resolve_initial_cap(
        mode=effective_mode,
        fixed_cap=max_items_per_user,
        memory_limit_mb=memory_limit_mb,
    )
    min_cap = 20
    safe_batch_size = max(200, int(batch_size))
    retained_neighbors = max(int(max_neighbors), int(max_neighbors) * max(1, int(retained_neighbors_factor)))

    users_total = len(user_items)
    logger.event(
        "STAGE1_CF_MODE",
        mode=effective_mode,
        memory_limit_mb=memory_limit_mb,
        initial_cap=current_cap,
        batch_size=safe_batch_size,
        retained_neighbors=retained_neighbors,
    )
    for i, items in enumerate(user_items.tolist(), start=1):
        uniq = list(dict.fromkeys(items))
        if current_cap > 0 and len(uniq) > current_cap:
            truncated_users += 1
            # Оставляем хвост истории, чтобы ограничить число пар и избежать OOM.
            uniq = uniq[-current_cap:]
        item_counts.update(uniq)
        for a, b in itertools.combinations(sorted(uniq), 2):
            neighbor_counts[a][b] += 1
            neighbor_counts[b][a] += 1

        if i % safe_batch_size == 0:
            _prune_neighbor_counts(neighbor_counts, retained_neighbors)

        if effective_mode == "auto" and memory_limit_mb is not None and i % 2000 == 0:
            rss_mb = _read_process_rss_mb()
            ratio = (rss_mb / memory_limit_mb) if memory_limit_mb > 0 else None
            if ratio is not None and ratio >= 0.82 and current_cap > min_cap:
                next_cap = max(min_cap, int(current_cap * 0.8))
                if next_cap < current_cap:
                    current_cap = next_cap
                    retained_neighbors = max(max_neighbors, int(retained_neighbors * 0.85))
                    _prune_neighbor_counts(neighbor_counts, retained_neighbors)
                    logger.event(
                        "STAGE1_CF_ADAPT",
                        done=i,
                        total=users_total,
                        rss_mb=rss_mb,
                        memory_limit_mb=memory_limit_mb,
                        memory_ratio=round(ratio, 4),
                        new_cap=current_cap,
                        retained_neighbors=retained_neighbors,
                    )

        if i % max(1, users_total // 20) == 0 or i == users_total:
            rss_mb = _read_process_rss_mb()
            ratio = (rss_mb / memory_limit_mb) if memory_limit_mb and memory_limit_mb > 0 else None
            logger.event(
                "STAGE1_CF_PROGRESS",
                done=i,
                total=users_total,
                truncated_users=truncated_users,
                max_items_per_user=current_cap,
                rss_mb=rss_mb,
                memory_limit_mb=memory_limit_mb,
                memory_ratio=(None if ratio is None else round(ratio, 4)),
                retained_neighbors=retained_neighbors,
            )

    _prune_neighbor_counts(neighbor_counts, retained_neighbors)

    out: dict[Any, list[tuple[Any, float]]] = {}
    for item_id, counts in neighbor_counts.items():
        vals: list[tuple[Any, float]] = []
        for other_id, co in counts.items():
            denom = math.sqrt(float(item_counts[item_id]) * float(item_counts[other_id]))
            if denom <= 0:
                continue
            vals.append((other_id, float(co / denom)))
        vals.sort(key=lambda x: x[1], reverse=True)
        out[item_id] = vals[:max_neighbors]
    return out


def _prune_neighbor_counts(neighbor_counts: dict[Any, Counter[Any]], retained_neighbors: int) -> None:
    if retained_neighbors <= 0:
        return
    for item_id, counts in list(neighbor_counts.items()):
        if len(counts) <= retained_neighbors:
            continue
        neighbor_counts[item_id] = Counter(dict(counts.most_common(retained_neighbors)))


def _resolve_initial_cap(mode: str, fixed_cap: int, memory_limit_mb: int | None) -> int:
    if mode == "fixed":
        return max(1, fixed_cap)
    if memory_limit_mb is None:
        return max(1, fixed_cap)
    if memory_limit_mb <= 8 * 1024:
        return 40
    if memory_limit_mb <= 16 * 1024:
        return 70
    if memory_limit_mb <= 32 * 1024:
        return 110
    return max(1, fixed_cap)


def _read_process_rss_mb() -> int:
    status_path = "/proc/self/status"
    try:
        with open(status_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("VmRSS:"):
                    continue
                value_kb = int(line.split()[1])
                return max(0, value_kb // 1024)
    except (OSError, ValueError, IndexError):
        return 0
    return 0


def _read_memory_limit_mb() -> int | None:
    candidates = [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
        except OSError:
            continue
        if not raw or raw.lower() == "max":
            continue
        try:
            value_bytes = int(raw)
        except ValueError:
            continue
        if value_bytes <= 0:
            continue
        # Некоторые среды возвращают "почти бесконечность" вместо реального лимита.
        if value_bytes >= 1 << 60:
            continue
        return value_bytes // (1024 * 1024)
    return None


def fit_content_neighbors(books: Any, max_neighbors: int, logger: Any) -> dict[Any, list[tuple[Any, float]]]:
    data = books.copy()
    for col in ["authors", "series", "tags"]:
        if col not in data.columns:
            data[col] = [[] for _ in range(len(data))]
        data[col] = data[col].apply(lambda x: x if isinstance(x, list) else [])

    by_item = data[["item_id", "authors", "series", "tags"]].drop_duplicates("item_id").reset_index(drop=True)

    author_index: dict[str, list[Any]] = defaultdict(list)
    series_index: dict[str, list[Any]] = defaultdict(list)
    tag_index: dict[str, list[Any]] = defaultdict(list)

    total = len(by_item)
    for i, row in enumerate(by_item.itertuples(index=False), start=1):
        item_id = row.item_id
        for a in list(dict.fromkeys(row.authors))[:8]:
            author_index[str(a)].append(item_id)
        for s in list(dict.fromkeys(row.series))[:4]:
            series_index[str(s)].append(item_id)
        for t in list(dict.fromkeys(row.tags))[:20]:
            tag_index[str(t)].append(item_id)
        if i % max(1, total // 20) == 0 or i == total:
            logger.event("STAGE1_CONTENT_INDEX_PROGRESS", done=i, total=total)

    out: dict[Any, list[tuple[Any, float]]] = {}
    for i, row in enumerate(by_item.itertuples(index=False), start=1):
        item_id = row.item_id
        score_map: dict[Any, float] = defaultdict(float)

        for a in list(dict.fromkeys(row.authors))[:8]:
            for other in author_index.get(str(a), []):
                if other != item_id:
                    score_map[other] += 2.0
        for s in list(dict.fromkeys(row.series))[:4]:
            for other in series_index.get(str(s), []):
                if other != item_id:
                    score_map[other] += 2.0
        for t in list(dict.fromkeys(row.tags))[:20]:
            for other in tag_index.get(str(t), []):
                if other != item_id:
                    score_map[other] += 0.5

        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:max_neighbors]
        out[item_id] = [(oid, float(score)) for oid, score in ranked]
        if i % max(1, total // 20) == 0 or i == total:
            logger.event("STAGE1_CONTENT_SCORE_PROGRESS", done=i, total=total)
    return out


def fit_metadata_retrieval(books: Any, logger: Any) -> dict[str, Any]:
    data = books.copy()
    for col in ["authors", "series", "tags"]:
        if col not in data.columns:
            data[col] = [[] for _ in range(len(data))]
        data[col] = data[col].apply(lambda x: x if isinstance(x, list) else [])

    by_item = data[["item_id", "authors", "series", "tags"]].drop_duplicates("item_id").reset_index(drop=True)
    author_index: dict[str, list[Any]] = defaultdict(list)
    series_index: dict[str, list[Any]] = defaultdict(list)
    tag_index: dict[str, list[Any]] = defaultdict(list)
    item_metadata: dict[Any, dict[str, list[str]]] = {}

    total = len(by_item)
    for i, row in enumerate(by_item.itertuples(index=False), start=1):
        item_id = row.item_id
        authors = [str(x) for x in list(dict.fromkeys(row.authors))[:8]]
        series = [str(x) for x in list(dict.fromkeys(row.series))[:4]]
        tags = [str(x) for x in list(dict.fromkeys(row.tags))[:20]]

        item_metadata[item_id] = {
            "authors": authors,
            "series": series,
            "tags": tags,
        }
        for value in authors:
            author_index[value].append(item_id)
        for value in series:
            series_index[value].append(item_id)
        for value in tags:
            tag_index[value].append(item_id)

        if i % max(1, total // 20) == 0 or i == total:
            logger.event("STAGE1_METADATA_INDEX_PROGRESS", done=i, total=total)

    return {
        "item_metadata": item_metadata,
        "author_index": dict(author_index),
        "series_index": dict(series_index),
        "tag_index": dict(tag_index),
    }
