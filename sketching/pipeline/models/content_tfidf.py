from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Any, Optional


logger = logging.getLogger(__name__)


class ContentTfidfRecommender:
    def __init__(self, max_features: int = 50000,
        min_df: int = 2, ngram_range: tuple[int, int] = (1, 2)) -> None:

        self.max_features = max_features
        self.min_df = min_df
        self.ngram_range = ngram_range

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.item_matrix = None
        self.item_ids: list[Any] = []
        self.item_id_to_idx: dict[Any, int] = {}
        self.popularity_fallback: list[Any] = []
        self.is_fitted = False

    def fit(self, item_text: pd.DataFrame,
            item_popularity: Optional[pd.DataFrame] = None) -> "ContentTfidfRecommender":

        if "item_id" not in item_text.columns or "item_text" not in item_text.columns:
            raise ValueError("item_text must contain item_id and item_text")

        data = item_text[["item_id", "item_text"]].drop_duplicates("item_id").reset_index(drop=True).copy()
        data["item_text"] = data["item_text"].fillna("").astype(str)

        self.item_ids = data["item_id"].tolist()
        self.item_id_to_idx = {item_id: i for i, item_id in enumerate(self.item_ids)}

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            ngram_range=self.ngram_range,
            lowercase=True,
        )
        self.item_matrix = self.vectorizer.fit_transform(data["item_text"].tolist())

        if item_popularity is not None and "item_id" in item_popularity.columns:
            self.popularity_fallback = item_popularity["item_id"].drop_duplicates().tolist()
        else:
            self.popularity_fallback = list(self.item_ids)

        self.is_fitted = True
        return self

    def score_user(self, seen_items: set[Any], top_n: int = 200) -> tuple[list[Any], list[float]]:
        if not self.is_fitted or self.item_matrix is None:
            raise ValueError("fit() must be called before score_user()")

        seen_idxs = [self.item_id_to_idx[x] for x in seen_items if x in self.item_id_to_idx]
        if not seen_idxs:
            return [], []

        profile = self.item_matrix[seen_idxs].mean(axis=0)
        scores = np.asarray(self.item_matrix.dot(profile.T)).ravel()
        if seen_idxs:
            scores[seen_idxs] = -np.inf

        valid_count = int(np.isfinite(scores).sum())
        if valid_count <= 0:
            return [], []

        top_n = min(top_n, valid_count)
        idx = np.argpartition(scores, -top_n)[-top_n:]
        idx = idx[np.argsort(scores[idx])[::-1]]

        items = [self.item_ids[i] for i in idx if np.isfinite(scores[i])]
        vals = [float(scores[i]) for i in idx if np.isfinite(scores[i])]
        return items, vals

    def recommend(self, user_ids: list[Any],
        seen_items_by_user: dict[Any, set[Any]], k: int = 10,
        candidate_top_n: int = 200) -> pd.DataFrame:

        if not self.is_fitted:
            raise ValueError("fit() must be called before recommend()")

        rows = []
        total = len(user_ids)
        step = max(1, total // 10) if total else 1
        for i, user_id in enumerate(user_ids, start=1):
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
            if total and (i == total or i % step == 0):
                logger.info("ContentTFIDF recommend: %s/%s", i, total)

        return pd.DataFrame(rows)
