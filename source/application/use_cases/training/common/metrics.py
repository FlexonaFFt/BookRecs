from __future__ import annotations

import math
from typing import Any


def recall_at_k(pred_items: list[Any], gt_items: list[Any], k: int) -> float:
    if not gt_items:
        return 0.0
    gt_set = set(gt_items)
    hits = sum(1 for x in pred_items[:k] if x in gt_set)
    denom = min(len(gt_set), k)
    return float(hits / denom) if denom > 0 else 0.0


def ndcg_at_k(pred_items: list[Any], gt_items: list[Any], k: int) -> float:
    if not gt_items:
        return 0.0
    gt_set = set(gt_items)
    dcg = 0.0
    for rank, item_id in enumerate(pred_items[:k], start=1):
        if item_id in gt_set:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_len = min(len(gt_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_len + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0
