from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from typing import Any


def fit_stage1(data: dict[str, Any], cmd: Any, logger: Any) -> dict[str, Any]:
    train = data["local_train"]
    books = data["books"]
    logger.start_step("stage1_fit", total=3)

    pop_items, pop_scores = fit_popularity(train)
    logger.progress("stage1_fit", done=1, total=3)

    cf_neighbors = fit_cf_neighbors(
        interactions=train,
        max_neighbors=cmd.cf_max_neighbors,
        logger=logger,
    )
    logger.progress("stage1_fit", done=2, total=3)

    content_similar = fit_content_neighbors(
        books=books,
        max_neighbors=cmd.content_max_neighbors,
        logger=logger,
    )
    logger.progress("stage1_fit", done=3, total=3)
    logger.end_step(
        "stage1_fit",
        status="SUCCESS",
        pop_items=len(pop_items),
        cf_items=len(cf_neighbors),
        content_items=len(content_similar),
    )
    return {
        "pop_items": pop_items,
        "pop_scores": pop_scores,
        "cf_neighbors": cf_neighbors,
        "content_similar": content_similar,
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


def fit_cf_neighbors(interactions: Any, max_neighbors: int, logger: Any) -> dict[Any, list[tuple[Any, float]]]:
    user_items = (
        interactions[["user_id", "item_id"]]
        .drop_duplicates(["user_id", "item_id"])
        .groupby("user_id", sort=False)["item_id"]
        .agg(list)
    )
    pair_counts: Counter[tuple[Any, Any]] = Counter()
    item_counts: Counter[Any] = Counter()

    users_total = len(user_items)
    for i, items in enumerate(user_items.tolist(), start=1):
        uniq = list(dict.fromkeys(items))
        item_counts.update(uniq)
        for a, b in itertools.combinations(sorted(uniq), 2):
            pair_counts[(a, b)] += 1
        if i % max(1, users_total // 20) == 0 or i == users_total:
            logger.event("STAGE1_CF_PROGRESS", done=i, total=users_total)

    neighbors: dict[Any, list[tuple[Any, float]]] = defaultdict(list)
    for (a, b), co in pair_counts.items():
        denom = math.sqrt(float(item_counts[a]) * float(item_counts[b]))
        if denom <= 0:
            continue
        score = float(co / denom)
        neighbors[a].append((b, score))
        neighbors[b].append((a, score))

    out: dict[Any, list[tuple[Any, float]]] = {}
    for item_id, vals in neighbors.items():
        vals.sort(key=lambda x: x[1], reverse=True)
        out[item_id] = vals[:max_neighbors]
    return out


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
