import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


logger = logging.getLogger(__name__)


# Форматировать время
def _fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m = seconds // 60
    s = seconds % 60
    if m > 0:
        return f"{m}м {s}с"
    return f"{s}с"


# Лог прогресса с ETA
def _log_progress(prefix: str, current: int, total: int, started_at: float, step: int) -> None:
    if total <= 0:
        return
    if current != total and current % step != 0:
        return
    elapsed = time.time() - started_at
    speed = current / elapsed if elapsed > 0 else 0.0
    eta = (total - current) / speed if speed > 0 else 0.0
    logger.info(
        "%s: %s/%s (%.1f%%), прошло=%s, осталось~%s",
        prefix, current, total, current * 100.0 / total, _fmt_seconds(elapsed), _fmt_seconds(eta)
    )


# Простая контентная модель на TF-IDF
class ContentTfidfRecommender:
    def __init__(
        self,
        max_features: int = 50000,
        min_df: int = 2,
        ngram_range: tuple = (1, 2),
        seed: int = 42,
    ) -> None:
        self.max_features = max_features
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.seed = seed

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.item_matrix = None
        self.item_ids: list = []
        self.item_id_to_idx: dict = {}
        self.popularity_fallback: list = []
        self.is_fitted = False

    # Обучить контентную модель по текстам айтемов
    def fit(self, item_text: pd.DataFrame, item_popularity: Optional[pd.DataFrame] = None) -> "ContentTfidfRecommender":
        started = time.time()
        if "item_id" not in item_text.columns or "item_text" not in item_text.columns:
            raise ValueError("item_text должен содержать колонки item_id и item_text")

        logger.info(
            "Обучение ContentTFIDF: items=%s, max_features=%s, min_df=%s, ngram=%s",
            len(item_text),
            self.max_features,
            self.min_df,
            self.ngram_range,
        )

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
            self.popularity_fallback = self.item_ids[:]

        self.is_fitted = True
        logger.info(
            "ContentTFIDF готов за %s: matrix=%s",
            _fmt_seconds(time.time() - started),
            getattr(self.item_matrix, "shape", None),
        )
        return self

    # Посчитать скоры кандидатов для одного пользователя
    def score_user(self, seen_items: set, top_n: int = 200) -> tuple[list, list]:
        if not self.is_fitted or self.item_matrix is None:
            raise ValueError("Сначала вызовите fit()")

        seen_idxs = [self.item_id_to_idx[x] for x in seen_items if x in self.item_id_to_idx]
        if len(seen_idxs) == 0:
            return [], []

        profile = self.item_matrix[seen_idxs].mean(axis=0)
        scores = np.asarray(self.item_matrix.dot(profile.T)).ravel()

        if len(seen_idxs) > 0:
            scores[seen_idxs] = -np.inf

        valid_count = np.isfinite(scores).sum()
        if valid_count == 0:
            return [], []

        top_n = min(top_n, int(valid_count))
        idx = np.argpartition(scores, -top_n)[-top_n:]
        idx = idx[np.argsort(scores[idx])[::-1]]

        items = [self.item_ids[i] for i in idx if np.isfinite(scores[i])]
        vals = [float(scores[i]) for i in idx if np.isfinite(scores[i])]
        return items, vals

    # Выдать рекомендации для списка пользователей
    def recommend(
        self,
        user_ids: list,
        *,
        seen_items_by_user: dict,
        k: int = 10,
        candidate_top_n: int = 200,
    ) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Сначала вызовите fit()")

        rows = []
        total = len(user_ids)
        step = max(100, total // 20 or 1)
        started = time.time()
        logger.info("Генерация рекомендаций ContentTFIDF: users=%s", total)

        for i, user_id in enumerate(user_ids, start=1):
            seen = seen_items_by_user.get(user_id, set())
            items, _scores = self.score_user(seen, top_n=max(candidate_top_n, k))

            recs = []
            for item_id in items:
                if item_id in seen:
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
            _log_progress("ContentTFIDF recommend", i, total, started, step)

        return pd.DataFrame(rows)
