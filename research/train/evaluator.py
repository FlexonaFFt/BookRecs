import math
from typing import Optional

import pandas as pd


# Проверить колонки
def _check_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: нет колонок {missing}. Есть: {list(df.columns)}")


# Подготовить таблицу предсказаний к grouped формату user_id -> pred_items
def ensure_grouped_predictions(
    predictions: pd.DataFrame,
    pred_col: str = "pred_items",
) -> pd.DataFrame:
    _check_columns(predictions, ["user_id"], "predictions")

    if pred_col in predictions.columns:
        return predictions[["user_id", pred_col]].copy()

    if "item_id" in predictions.columns:
        return (
            predictions.groupby("user_id", sort=False)["item_id"]
            .apply(list)
            .reset_index()
            .rename(columns={"item_id": pred_col})
        )

    raise ValueError("predictions должны содержать pred_items или item_id")


# Слить gt и predictions в единую таблицу
def build_eval_table(
    eval_ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    pred_col: str = "pred_items",
) -> pd.DataFrame:
    _check_columns(eval_ground_truth, ["user_id", "item_id"], "eval_ground_truth")
    preds = ensure_grouped_predictions(predictions, pred_col=pred_col)
    merged = eval_ground_truth.merge(preds, on="user_id", how="left")
    merged[pred_col] = merged[pred_col].apply(lambda x: x if isinstance(x, list) else [])
    merged["item_id"] = merged["item_id"].apply(lambda x: x if isinstance(x, list) else [])
    return merged


# Рассчитать dcg@k для бинарной релевантности
def _dcg_at_k(pred_items: list, gt_items_set: set, k: int) -> float:
    score = 0.0
    for rank, item_id in enumerate(pred_items[:k], start=1):
        if item_id in gt_items_set:
            score += 1.0 / math.log2(rank + 1)
    return score


# Рассчитать ndcg@k и recall@k по сегменту
def _segment_ranking_metrics(
    eval_table: pd.DataFrame,
    *,
    gt_col: str = "item_id",
    pred_col: str = "pred_items",
    k: int = 10,
) -> dict:
    ndcg_scores = []
    recall_scores = []
    n_users = 0

    for _, row in eval_table.iterrows():
        gt_items = row[gt_col] if isinstance(row[gt_col], list) else []
        pred_items = row[pred_col] if isinstance(row[pred_col], list) else []
        if len(gt_items) == 0:
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
        return {"users": 0, f"ndcg@{k}": 0.0, f"recall@{k}": 0.0}

    return {
        "users": n_users,
        f"ndcg@{k}": float(sum(ndcg_scores) / n_users),
        f"recall@{k}": float(sum(recall_scores) / n_users),
    }


# Рассчитать coverage@k
def coverage_at_k(
    predictions: pd.DataFrame,
    *,
    catalog_size: int,
    pred_col: str = "pred_items",
    k: int = 10,
) -> float:
    preds = ensure_grouped_predictions(predictions, pred_col=pred_col)
    uniq = set()
    for items in preds[pred_col]:
        if isinstance(items, list):
            uniq.update(items[:k])
    if catalog_size <= 0:
        return 0.0
    return float(len(uniq) / catalog_size)


# Оставить в gt только items из выбранного сегмента
def filter_ground_truth_by_items(
    eval_ground_truth: pd.DataFrame,
    item_ids: set,
) -> pd.DataFrame:
    out = eval_ground_truth.copy()
    out["item_id"] = out["item_id"].apply(
        lambda xs: [x for x in xs if x in item_ids] if isinstance(xs, list) else []
    )
    return out


# Полная оценка: overall + warm + cold + coverage
def evaluate_predictions(
    eval_ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    catalog_size: int,
    warm_item_ids: Optional[set] = None,
    cold_item_ids: Optional[set] = None,
    pred_col: str = "pred_items",
    k: int = 10,
) -> dict:
    overall_table = build_eval_table(eval_ground_truth, predictions, pred_col=pred_col)
    overall_metrics = _segment_ranking_metrics(overall_table, gt_col="item_id", pred_col=pred_col, k=k)

    metrics = {
        f"ndcg@{k}": overall_metrics[f"ndcg@{k}"],
        f"recall@{k}": overall_metrics[f"recall@{k}"],
        f"coverage@{k}": coverage_at_k(predictions, catalog_size=catalog_size, pred_col=pred_col, k=k),
        "eval_users": overall_metrics["users"],
    }

    if warm_item_ids is not None:
        gt_warm = filter_ground_truth_by_items(eval_ground_truth, warm_item_ids)
        warm_table = build_eval_table(gt_warm, predictions, pred_col=pred_col)
        warm_metrics = _segment_ranking_metrics(warm_table, gt_col="item_id", pred_col=pred_col, k=k)
        metrics[f"warm_ndcg@{k}"] = warm_metrics[f"ndcg@{k}"]
        metrics[f"warm_recall@{k}"] = warm_metrics[f"recall@{k}"]
        metrics["warm_eval_users"] = warm_metrics["users"]

    if cold_item_ids is not None:
        gt_cold = filter_ground_truth_by_items(eval_ground_truth, cold_item_ids)
        cold_table = build_eval_table(gt_cold, predictions, pred_col=pred_col)
        cold_metrics = _segment_ranking_metrics(cold_table, gt_col="item_id", pred_col=pred_col, k=k)
        metrics[f"cold_ndcg@{k}"] = cold_metrics[f"ndcg@{k}"]
        metrics[f"cold_recall@{k}"] = cold_metrics[f"recall@{k}"]
        metrics["cold_eval_users"] = cold_metrics["users"]

    return metrics
