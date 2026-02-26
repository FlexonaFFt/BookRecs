from __future__ import annotations

import math
from typing import Any, Optional

import pandas as pd


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}; got {list(df.columns)}")


def _to_items_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return []


def ensure_grouped_predictions(predictions: pd.DataFrame, pred_col: str = "pred_items") -> pd.DataFrame:
    _require_columns(predictions, ["user_id"], "predictions")

    if pred_col in predictions.columns:
        return predictions[["user_id", pred_col]].copy()

    if "item_id" in predictions.columns:
        return (
            predictions.groupby("user_id", sort=False)["item_id"]
            .apply(list)
            .reset_index()
            .rename(columns={"item_id": pred_col})
        )

    raise ValueError("predictions must contain pred_items or item_id")


def build_eval_table(
    eval_ground_truth: pd.DataFrame, predictions: pd.DataFrame,
    pred_col: str = "pred_items") -> pd.DataFrame:

    _require_columns(eval_ground_truth, ["user_id", "item_id"], "eval_ground_truth")
    preds = ensure_grouped_predictions(predictions, pred_col=pred_col)
    merged = eval_ground_truth.merge(preds, on="user_id", how="left")
    merged[pred_col] = merged[pred_col].apply(_to_items_list)
    merged["item_id"] = merged["item_id"].apply(_to_items_list)
    return merged


def _dcg_at_k(pred_items: list[Any], gt_items_set: set[Any], k: int) -> float:
    score = 0.0
    for rank, item_id in enumerate(pred_items[:k], start=1):
        if item_id in gt_items_set:
            score += 1.0 / math.log2(rank + 1)
    return score


def _segment_ranking_metrics(
    eval_table: pd.DataFrame, gt_col: str = "item_id",
    pred_col: str = "pred_items", k: int = 10) -> dict[str, float]:

    ndcg_scores = []
    recall_scores = []
    n_users = 0

    for _, row in eval_table.iterrows():
        gt_items = row[gt_col] if isinstance(row[gt_col], list) else []
        pred_items = row[pred_col] if isinstance(row[pred_col], list) else []
        if not gt_items:
            continue

        n_users += 1
        gt_set = set(gt_items)
        hits = sum(1 for x in pred_items[:k] if x in gt_set)

        ideal_len = min(len(gt_items), k)
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_len + 1))
        dcg = _dcg_at_k(pred_items, gt_set, k)

        recall_scores.append(hits / ideal_len if ideal_len > 0 else 0.0)
        ndcg_scores.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)

    if n_users == 0:
        return {"users": 0.0, f"ndcg@{k}": 0.0, f"recall@{k}": 0.0}

    return {
        "users": float(n_users),
        f"ndcg@{k}": float(sum(ndcg_scores) / n_users),
        f"recall@{k}": float(sum(recall_scores) / n_users),
    }


def coverage_at_k(
    predictions: pd.DataFrame, catalog_size: int,
    pred_col: str = "pred_items", k: int = 10) -> float:

    preds = ensure_grouped_predictions(predictions, pred_col=pred_col)
    uniq: set[Any] = set()
    for items in preds[pred_col]:
        if isinstance(items, list):
            uniq.update(items[:k])
    if catalog_size <= 0:
        return 0.0
    return float(len(uniq) / catalog_size)


def filter_ground_truth_by_items(eval_ground_truth: pd.DataFrame, item_ids: set[Any]) -> pd.DataFrame:
    out = eval_ground_truth.copy()
    out["item_id"] = out["item_id"].apply(lambda xs: [x for x in _to_items_list(xs) if x in item_ids])
    return out


def evaluate_predictions(
    eval_ground_truth: pd.DataFrame, predictions: pd.DataFrame, catalog_size: int,
    warm_item_ids: Optional[set[Any]] = None, cold_item_ids: Optional[set[Any]] = None,
    pred_col: str = "pred_items", k: int = 10) -> dict[str, float]:

    overall_table = build_eval_table(eval_ground_truth, predictions, pred_col=pred_col)
    overall = _segment_ranking_metrics(overall_table, gt_col="item_id", pred_col=pred_col, k=k)

    metrics: dict[str, float] = {
        f"ndcg@{k}": float(overall[f"ndcg@{k}"]),
        f"recall@{k}": float(overall[f"recall@{k}"]),
        f"coverage@{k}": float(coverage_at_k(predictions, catalog_size=catalog_size, pred_col=pred_col, k=k)),
        "eval_users": float(overall["users"]),
    }

    if warm_item_ids is not None:
        gt_warm = filter_ground_truth_by_items(eval_ground_truth, warm_item_ids)
        warm_table = build_eval_table(gt_warm, predictions, pred_col=pred_col)
        warm = _segment_ranking_metrics(warm_table, gt_col="item_id", pred_col=pred_col, k=k)
        metrics[f"warm_ndcg@{k}"] = float(warm[f"ndcg@{k}"])
        metrics[f"warm_recall@{k}"] = float(warm[f"recall@{k}"])
        metrics["warm_eval_users"] = float(warm["users"])

    if cold_item_ids is not None:
        gt_cold = filter_ground_truth_by_items(eval_ground_truth, cold_item_ids)
        cold_table = build_eval_table(gt_cold, predictions, pred_col=pred_col)
        cold = _segment_ranking_metrics(cold_table, gt_col="item_id", pred_col=pred_col, k=k)
        metrics[f"cold_ndcg@{k}"] = float(cold[f"ndcg@{k}"])
        metrics[f"cold_recall@{k}"] = float(cold[f"recall@{k}"])
        metrics["cold_eval_users"] = float(cold["users"])

    return metrics
