import logging
import time
from typing import Optional

import pandas as pd

try:
    from content_model import ContentTfidfRecommender
except ImportError:
    from research.train.content_model import ContentTfidfRecommender  # type: ignore


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


# Гибридная модель content + popularity
class HybridContentPopularRecommender:
    def __init__(
        self,
        content_weight: float = 0.8,
        popularity_weight: float = 0.2,
        content_top_n: int = 200,
        pop_candidates_n: int = 200,
        max_features: int = 50000,
        min_df: int = 2,
    ) -> None:
        self.content_weight = content_weight
        self.popularity_weight = popularity_weight
        self.content_top_n = content_top_n
        self.pop_candidates_n = pop_candidates_n

        self.content_model = ContentTfidfRecommender(max_features=max_features, min_df=min_df)
        self.pop_items: list = []
        self.pop_scores: dict = {}
        self.is_fitted = False

    # Обучить гибридную модель
    def fit(self, item_text: pd.DataFrame, item_popularity: pd.DataFrame) -> "HybridContentPopularRecommender":
        started = time.time()
        if "item_id" not in item_popularity.columns:
            raise ValueError("item_popularity должен содержать item_id")

        logger.info(
            "Обучение Hybrid content+popular (cw=%.2f, pw=%.2f)",
            self.content_weight,
            self.popularity_weight,
        )
        self.content_model.fit(item_text=item_text, item_popularity=item_popularity)

        pop = item_popularity.copy()
        if "popularity_weight" not in pop.columns:
            pop["popularity_weight"] = 1.0
        pop = pop[["item_id", "popularity_weight"]].drop_duplicates("item_id")
        max_pop = float(pop["popularity_weight"].max()) if len(pop) else 1.0
        if max_pop <= 0:
            max_pop = 1.0
        pop["pop_norm"] = pop["popularity_weight"] / max_pop
        self.pop_items = pop.sort_values("pop_norm", ascending=False)["item_id"].tolist()
        self.pop_scores = dict(zip(pop["item_id"], pop["pop_norm"]))

        self.is_fitted = True
        logger.info("Hybrid готов за %s", _fmt_seconds(time.time() - started))
        return self

    # Выдать рекомендации для списка пользователей
    def recommend(
        self,
        user_ids: list,
        *,
        seen_items_by_user: dict,
        k: int = 10,
    ) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Сначала вызовите fit()")

        rows = []
        total = len(user_ids)
        step = max(100, total // 20 or 1)
        started = time.time()
        logger.info("Генерация рекомендаций Hybrid: users=%s", total)

        pop_candidate_items = self.pop_items[: self.pop_candidates_n]

        for i, user_id in enumerate(user_ids, start=1):
            seen = seen_items_by_user.get(user_id, set())
            content_items, content_scores = self.content_model.score_user(seen, top_n=self.content_top_n)

            score_map = {}
            if content_scores:
                max_content = max(content_scores)
                min_content = min(content_scores)
                denom = (max_content - min_content) if max_content != min_content else 1.0
                for item_id, s in zip(content_items, content_scores):
                    score_map[item_id] = self.content_weight * ((s - min_content) / denom)

            for item_id in pop_candidate_items:
                score_map[item_id] = score_map.get(item_id, 0.0) + self.popularity_weight * float(self.pop_scores.get(item_id, 0.0))

            ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
            recs = []
            for item_id, _ in ranked:
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
            _log_progress("Hybrid recommend", i, total, started, step)

        return pd.DataFrame(rows)
