from __future__ import annotations

from typing import Any, Optional
import pandas as pd


class TopPopularRecommender:
    def __init__(self) -> None:
        self.top_items: list[Any] = []
        self.is_fitted = False

    def fit(self, local_train: pd.DataFrame) -> "TopPopularRecommender":
        if "item_id" not in local_train.columns:
            raise ValueError("local_train must contain item_id")

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
                .reset_index(drop=True)
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
                .reset_index(drop=True)
            )

        self.top_items = popularity["item_id"].tolist()
        self.is_fitted = True
        return self

    def recommend(self, user_ids: list[Any],
        seen_items_by_user: Optional[dict[Any, set[Any]]] = None, k: int = 10) -> pd.DataFrame:

        if not self.is_fitted:
            raise ValueError("fit() must be called before recommend()")

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
