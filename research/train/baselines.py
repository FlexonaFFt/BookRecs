import numpy as np
import pandas as pd
from typing import Optional


# TopPopular baseline для сравнения
class TopPopularRecommender:
    def __init__(self) -> None:
        self.top_items: list = []
        self.is_fitted = False

    # Обучить baseline по train
    def fit(self, local_train: pd.DataFrame) -> "TopPopularRecommender":
        if "item_id" not in local_train.columns:
            raise ValueError("local_train должен содержать item_id")

        if "interaction_weight" in local_train.columns:
            popularity = (
                local_train.groupby("item_id", as_index=False)
                .agg(
                    n_interactions=("item_id", "size"),
                    popularity_weight=("interaction_weight", "sum"),
                )
                .sort_values(
                    ["popularity_weight", "n_interactions", "item_id"],
                    ascending=[False, False, True],
                )
            )
        else:
            popularity = (
                local_train.groupby("item_id", as_index=False)
                .agg(n_interactions=("item_id", "size"))
                .assign(popularity_weight=lambda x: x["n_interactions"].astype(float))
                .sort_values(
                    ["popularity_weight", "n_interactions", "item_id"],
                    ascending=[False, False, True],
                )
            )

        self.top_items = popularity["item_id"].tolist()
        self.is_fitted = True
        return self

    # Выдать top-k рекомендаций пользователям
    def recommend(
        self,
        user_ids: list,
        *,
        seen_items_by_user: Optional[dict] = None,
        k: int = 10,
    ) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Сначала вызовите fit()")
        if seen_items_by_user is None:
            seen_items_by_user = {}

        rows = []
        for user_id in user_ids:
            seen = seen_items_by_user.get(user_id, set())
            recs = []
            for item_id in self.top_items:
                if item_id in seen:
                    continue
                recs.append(item_id)
                if len(recs) >= k:
                    break
            rows.append({"user_id": user_id, "pred_items": recs})
        return pd.DataFrame(rows)


# Random baseline только для sanity-check
class RandomRecommender:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.candidate_item_ids: list = []
        self.is_fitted = False

    # Сохранить пул кандидатов
    def fit(self, candidate_item_ids: list) -> "RandomRecommender":
        self.candidate_item_ids = list(candidate_item_ids)
        self.is_fitted = True
        return self

    # Выдать случайные рекомендации
    def recommend(
        self,
        user_ids: list,
        *,
        seen_items_by_user: Optional[dict] = None,
        k: int = 10,
    ) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Сначала вызовите fit()")
        if seen_items_by_user is None:
            seen_items_by_user = {}

        rows = []
        for user_id in user_ids:
            seen = seen_items_by_user.get(user_id, set())
            pool = [x for x in self.candidate_item_ids if x not in seen]
            if len(pool) <= k:
                recs = pool
            else:
                idx = self.rng.choice(len(pool), size=k, replace=False)
                recs = [pool[i] for i in idx]
            rows.append({"user_id": user_id, "pred_items": recs})
        return pd.DataFrame(rows)
