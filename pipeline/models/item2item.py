from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from math import sqrt
from typing import Any, Optional

import pandas as pd


class Item2ItemRecommender:
    def __init__(self, min_cooccurrence: int = 2,
        max_neighbors: int = 200, max_user_items: int = 50,
        shrinkage: float = 10.0) -> None:

        self.min_cooccurrence = min_cooccurrence
        self.max_neighbors = max_neighbors
        self.max_user_items = max_user_items
        self.shrinkage = shrinkage

        self.item_user_counts: dict[Any, int] = {}
        self.neighbors: dict[Any, list[tuple[Any, float]]] = {}
        self.popularity_fallback: list[Any] = []
        self.is_fitted = False

    def fit(self, local_train: pd.DataFrame,
        item_popularity: Optional[pd.DataFrame] = None) -> "Item2ItemRecommender":

        if "user_id" not in local_train.columns or "item_id" not in local_train.columns:
            raise ValueError("local_train must contain user_id and item_id")

        user_items = (
            local_train[["user_id", "item_id"]]
            .drop_duplicates(["user_id", "item_id"])
            .groupby("user_id", sort=False)["item_id"]
            .agg(list)
        )

        item_user_counts_counter: Counter[Any] = Counter()
        pair_counts: Counter[tuple[Any, Any]] = Counter()

        for items in user_items.tolist():
            if not items:
                continue
            uniq_items = list(dict.fromkeys(items))
            if len(uniq_items) > self.max_user_items:
                uniq_items = uniq_items[: self.max_user_items]

            item_user_counts_counter.update(uniq_items)

            if len(uniq_items) < 2:
                continue

            for a, b in combinations(uniq_items, 2):
                if a == b:
                    continue
                if a < b:
                    pair_counts[(a, b)] += 1
                else:
                    pair_counts[(b, a)] += 1

        self.item_user_counts = dict(item_user_counts_counter)

        neighbor_map: dict[Any, list[tuple[Any, float]]] = defaultdict(list)
        for (a, b), co in pair_counts.items():
            if co < self.min_cooccurrence:
                continue

            cnt_a = self.item_user_counts.get(a, 0)
            cnt_b = self.item_user_counts.get(b, 0)
            if cnt_a <= 0 or cnt_b <= 0:
                continue

            cosine = co / sqrt(cnt_a * cnt_b)
            score = cosine * (co / (co + self.shrinkage))

            if score <= 0:
                continue

            neighbor_map[a].append((b, float(score)))
            neighbor_map[b].append((a, float(score)))

        self.neighbors = {}
        for item_id, vals in neighbor_map.items():
            vals.sort(key=lambda x: x[1], reverse=True)
            self.neighbors[item_id] = vals[: self.max_neighbors]

        if item_popularity is not None and "item_id" in item_popularity.columns:
            self.popularity_fallback = item_popularity["item_id"].drop_duplicates().tolist()
        else:
            self.popularity_fallback = [
                item_id
                for item_id, _ in sorted(
                    self.item_user_counts.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ]

        self.is_fitted = True
        return self

    def score_user(self, seen_items: set[Any], top_n: int = 200) -> tuple[list[Any], list[float]]:
        if not self.is_fitted:
            raise ValueError("fit() must be called before score_user()")

        if not seen_items:
            return [], []

        score_map: dict[Any, float] = {}
        for item_id in seen_items:
            for neighbor_id, score in self.neighbors.get(item_id, []):
                if neighbor_id in seen_items:
                    continue
                score_map[neighbor_id] = score_map.get(neighbor_id, 0.0) + float(score)

        if not score_map:
            return [], []

        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:top_n]
        items = [item_id for item_id, _ in ranked]
        scores = [float(score) for _, score in ranked]
        return items, scores

    def recommend(self, user_ids: list[Any],
        seen_items_by_user: dict[Any, set[Any]], k: int = 10,
        candidate_top_n: int = 200) -> pd.DataFrame:

        if not self.is_fitted:
            raise ValueError("fit() must be called before recommend()")

        rows = []
        for user_id in user_ids:
            seen = seen_items_by_user.get(user_id, set())
            items, _ = self.score_user(seen, top_n=max(candidate_top_n, k))

            recs: list[Any] = []
            for item_id in items:
                if item_id in seen or item_id in recs:
                    continue
                recs.append(item_id)
                if len(recs) >= k:
                    break

            if len(recs) < k:
                for item_id in self.popularity_fallback:
                    if item_id in seen or item_id in recs:
                        continue
                    recs.append(item_id)
                    if len(recs) >= k:
                        break

            rows.append({"user_id": user_id, "pred_items": recs})

        return pd.DataFrame(rows)
