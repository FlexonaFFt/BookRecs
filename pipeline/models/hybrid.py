from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from models.content_tfidf import ContentTfidfRecommender
from models.item2item import Item2ItemRecommender


logger = logging.getLogger(__name__)


class HybridRecommender:
    def __init__(
        self, cf_weight: float = 0.5,
        content_weight: float = 0.35,
        pop_weight: float = 0.15,

        cf_top_n: int = 300, content_top_n: int = 300,
        pop_top_n: int = 300, content_max_features: int = 50000,
        content_min_df: int = 2,

        cf_min_cooccurrence: int = 2, cf_max_neighbors: int = 200,
        cf_max_user_items: int = 50, cf_shrinkage: float = 10.0) -> None:

        self.cf_weight = cf_weight
        self.content_weight = content_weight
        self.pop_weight = pop_weight
        self.cf_top_n = cf_top_n
        self.content_top_n = content_top_n
        self.pop_top_n = pop_top_n

        self.cf_model = Item2ItemRecommender(
            min_cooccurrence=cf_min_cooccurrence,
            max_neighbors=cf_max_neighbors,
            max_user_items=cf_max_user_items,
            shrinkage=cf_shrinkage,
        )
        self.content_model = ContentTfidfRecommender(
            max_features=content_max_features,
            min_df=content_min_df,
        )
        self.pop_items: list[Any] = []
        self.pop_scores: dict[Any, float] = {}
        self.is_fitted = False

    def fit(self, local_train: pd.DataFrame,
        item_text: pd.DataFrame, item_popularity: pd.DataFrame) -> "HybridRecommender":

        self.cf_model.fit(local_train=local_train, item_popularity=item_popularity)
        self.content_model.fit(item_text=item_text, item_popularity=item_popularity)

        if "item_id" not in item_popularity.columns:
            raise ValueError("item_popularity must contain item_id")

        pop = item_popularity.copy()
        if "popularity_weight" not in pop.columns:
            if "n_interactions" in pop.columns:
                pop["popularity_weight"] = pop["n_interactions"].astype(float)
            else:
                pop["popularity_weight"] = 1.0

        pop = pop[["item_id", "popularity_weight"]].drop_duplicates("item_id")
        max_pop = float(pop["popularity_weight"].max()) if len(pop) else 1.0
        if max_pop <= 0:
            max_pop = 1.0
        pop["pop_norm"] = pop["popularity_weight"] / max_pop
        pop = pop.sort_values(["pop_norm", "item_id"], ascending=[False, True]).reset_index(drop=True)

        self.pop_items = pop["item_id"].tolist()
        self.pop_scores = dict(zip(pop["item_id"], pop["pop_norm"].astype(float)))
        self.is_fitted = True
        return self

    def _normalize_scores(self, items: list[Any], scores: list[float]) -> dict[Any, float]:

        if not items or not scores:
            return {}
        mx = max(scores)
        mn = min(scores)
        denom = (mx - mn) if mx != mn else 1.0
        return {item_id: float((score - mn) / denom) for item_id, score in zip(items, scores)}

    def score_user(self, seen_items: set[Any], top_n: int = 300) -> tuple[list[Any], list[float]]:

        if not self.is_fitted:
            raise ValueError("fit() must be called before score_user()")

        cf_items, cf_scores = self.cf_model.score_user(seen_items, top_n=self.cf_top_n)
        content_items, content_scores = self.content_model.score_user(seen_items, top_n=self.content_top_n)

        cf_map = self._normalize_scores(cf_items, cf_scores)
        content_map = self._normalize_scores(content_items, content_scores)

        score_map: dict[Any, float] = {}

        for item_id, score in cf_map.items():
            score_map[item_id] = score_map.get(item_id, 0.0) + self.cf_weight * float(score)

        for item_id, score in content_map.items():
            score_map[item_id] = score_map.get(item_id, 0.0) + self.content_weight * float(score)

        for item_id in self.pop_items[: self.pop_top_n]:
            score_map[item_id] = score_map.get(item_id, 0.0) + self.pop_weight * float(self.pop_scores.get(item_id, 0.0))

        if seen_items:
            for item_id in seen_items:
                score_map.pop(item_id, None)

        if not score_map:
            return [], []

        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:top_n]
        items = [item_id for item_id, _ in ranked]
        scores = [float(score) for _, score in ranked]
        return items, scores

    def recommend(
        self, user_ids: list[Any],
        seen_items_by_user: dict[Any, set[Any]], k: int = 10) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("fit() must be called before recommend()")

        rows = []
        candidate_top_n = max(k, self.cf_top_n, self.content_top_n, self.pop_top_n)
        total = len(user_ids)
        step = max(1, total // 10) if total else 1

        for i, user_id in enumerate(user_ids, start=1):
            seen = seen_items_by_user.get(user_id, set())
            items, _ = self.score_user(seen_items=seen, top_n=candidate_top_n)

            recs: list[Any] = []
            for item_id in items:
                if item_id in seen or item_id in recs:
                    continue
                recs.append(item_id)
                if len(recs) >= k:
                    break

            if len(recs) < k:
                for item_id in self.pop_items:
                    if item_id in seen or item_id in recs:
                        continue
                    recs.append(item_id)
                    if len(recs) >= k:
                        break

            rows.append({"user_id": user_id, "pred_items": recs})
            if total and (i == total or i % step == 0):
                logger.info("Hybrid recommend: %s/%s", i, total)

        return pd.DataFrame(rows)
