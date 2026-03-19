from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

try:
    from catboost import CatBoostClassifier
except ModuleNotFoundError:
    CatBoostClassifier = None

from source.application.ports import PreRankerPort
from source.domain.entities import Candidate, ScoredCandidate
from source.infrastructure.ranking.prerank.feature_builder import (
    PRE_RANK_FEATURE_COLUMNS,
    FeatureBuilder,
)


@dataclass(frozen=True)
class CatBoostPreRankerConfig:
    iterations: int = 250
    depth: int = 6
    learning_rate: float = 0.08
    loss_function: str = "Logloss"
    eval_metric: str = "AUC"
    random_seed: int = 42
    verbose: bool = False
    auto_class_weights: str = "Balanced"
    allow_writing_files: bool = False
    feature_names: list[str] = field(
        default_factory=lambda: list(PRE_RANK_FEATURE_COLUMNS)
    )


class CatBoostPreRanker(PreRankerPort):
    """Candidate-level pre-ranker backed by CatBoostClassifier."""

    def __init__(
        self,
        model: Any,
        cfg: CatBoostPreRankerConfig | None = None,
        feature_builder: FeatureBuilder | None = None,
    ) -> None:
        self._model = model
        self._cfg = cfg or CatBoostPreRankerConfig()
        self._feature_builder = feature_builder or FeatureBuilder()

    @property
    def cfg(self) -> CatBoostPreRankerConfig:
        return self._cfg

    @classmethod
    def fit(
        cls,
        *,
        train_rows: list[dict[str, Any]],
        eval_rows: list[dict[str, Any]] | None = None,
        cfg: CatBoostPreRankerConfig | None = None,
    ) -> "CatBoostPreRanker":
        if pd is None:
            raise RuntimeError("pandas is required for CatBoost pre-ranker training.")
        if CatBoostClassifier is None:
            raise RuntimeError("catboost is required for CatBoost pre-ranker training.")
        if not train_rows:
            raise ValueError("train_rows must not be empty")

        cfg = cfg or CatBoostPreRankerConfig()
        feature_names = list(cfg.feature_names)
        train_df = pd.DataFrame(train_rows)
        x_train = train_df[feature_names]
        y_train = train_df["label"]

        model = CatBoostClassifier(
            iterations=cfg.iterations,
            depth=cfg.depth,
            learning_rate=cfg.learning_rate,
            loss_function=cfg.loss_function,
            eval_metric=cfg.eval_metric,
            random_seed=cfg.random_seed,
            verbose=cfg.verbose,
            auto_class_weights=cfg.auto_class_weights,
            allow_writing_files=cfg.allow_writing_files,
        )

        fit_kwargs: dict[str, Any] = {}
        if eval_rows:
            eval_df = pd.DataFrame(eval_rows)
            fit_kwargs["eval_set"] = (eval_df[feature_names], eval_df["label"])
            fit_kwargs["use_best_model"] = True
        model.fit(x_train, y_train, **fit_kwargs)
        return cls(model=model, cfg=cfg)

    def rank(
        self,
        candidates: list[Candidate],
        user_id: Any,
        history_len: int,
        cold_item_ids: set[Any],
        top_m: int,
    ) -> list[ScoredCandidate]:
        if top_m <= 0:
            return []
        rows = self._feature_builder.build(
            candidates=candidates,
            user_id=user_id,
            history_len=history_len,
            cold_item_ids=cold_item_ids,
        )
        if not rows:
            return []

        feature_names = list(self._cfg.feature_names)
        table = [
            {key: row.features.get(key, 0.0) for key in feature_names} for row in rows
        ]
        scores = self._predict_scores(table)

        scored: list[ScoredCandidate] = []
        for row, pre_score in zip(rows, scores):
            scored.append(
                ScoredCandidate(
                    user_id=row.user_id,
                    item_id=row.item_id,
                    source=row.source,
                    base_score=row.base_score,
                    pre_score=float(pre_score),
                    features=row.features,
                )
            )
        scored.sort(key=lambda x: x.pre_score, reverse=True)
        return scored[:top_m]

    def _predict_scores(self, rows: list[dict[str, float]]) -> list[float]:
        if pd is not None:
            features = pd.DataFrame(rows)
        else:
            features = rows

        if hasattr(self._model, "predict_proba"):
            pred = self._model.predict_proba(features)
            return [float(row[1]) for row in pred]
        if hasattr(self._model, "predict"):
            pred = self._model.predict(features)
            return [float(value) for value in pred]
        raise RuntimeError(
            "CatBoost pre-ranker model does not support predict_proba/predict"
        )
