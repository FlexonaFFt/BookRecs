from __future__ import annotations

import pytest

from source.application.use_cases.training.common.metrics import ndcg_at_k, recall_at_k


def test_recall_at_k_returns_zero_for_empty_gt() -> None:
    assert recall_at_k(pred_items=[1, 2, 3], gt_items=[], k=3) == 0.0


def test_recall_at_k_computes_hits_over_min_len_and_k() -> None:
    # hits in top-3: {2, 4} -> 2 hits; denom=min(3,3)=3
    value = recall_at_k(pred_items=[2, 9, 4, 7], gt_items=[1, 2, 4], k=3)
    assert value == pytest.approx(2 / 3)


def test_ndcg_at_k_returns_zero_for_empty_gt() -> None:
    assert ndcg_at_k(pred_items=[1, 2], gt_items=[], k=2) == 0.0


def test_ndcg_at_k_is_one_for_ideal_order() -> None:
    value = ndcg_at_k(pred_items=[1, 2, 3], gt_items=[1, 2, 3], k=3)
    assert value == pytest.approx(1.0)


def test_ndcg_at_k_less_than_one_for_non_ideal_order() -> None:
    ideal = ndcg_at_k(pred_items=[1, 2, 3], gt_items=[1, 2], k=3)
    shuffled = ndcg_at_k(pred_items=[3, 1, 2], gt_items=[1, 2], k=3)
    assert shuffled < ideal
    assert 0.0 <= shuffled <= 1.0
