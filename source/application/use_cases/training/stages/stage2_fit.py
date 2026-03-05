from __future__ import annotations

from dataclasses import asdict
from typing import Any

from source.application.use_cases.ranking.generate_candidates import (
    GenerateCandidatesCommand,
    GenerateCandidatesUseCase,
)
from source.application.use_cases.ranking.prerank_candidates import (
    PreRankCandidatesCommand,
    PreRankCandidatesUseCase,
)
from source.application.use_cases.training.common.data_ops import build_seen_map, build_val_ground_truth, cold_items
from source.infrastructure.ranking.candidates import (
    CfCandidateSource,
    ContentCandidateSource,
    PopularCandidateSource,
)
from source.infrastructure.ranking.prerank import LinearPreRanker, LinearPreRankerConfig


def fit_stage2(data: dict[str, Any], stage1: dict[str, Any], cmd: Any, logger: Any) -> LinearPreRanker:
    logger.start_step("stage2_fit", total=1)
    val_users, gt_map = build_val_ground_truth(data["local_val"], limit=cmd.eval_users_limit)
    seen_by_user = build_seen_map(data["local_train"])
    cold = cold_items(data["local_train"], data["local_val"])

    candidate_sources = [
        CfCandidateSource(stage1["cf_neighbors"]),
        ContentCandidateSource(stage1["content_similar"]),
        PopularCandidateSource(stage1["pop_items"], stage1["pop_scores"]),
    ]
    stage1_uc = GenerateCandidatesUseCase(sources=candidate_sources, fallback_source=candidate_sources[-1])

    grid = [
        LinearPreRankerConfig(w_base=1.0, w_cf=0.2, w_content=0.2, w_pop=0.1, w_cold=0.1, w_history=0.02),
        LinearPreRankerConfig(w_base=1.0, w_cf=0.3, w_content=0.2, w_pop=0.1, w_cold=0.1, w_history=0.02),
        LinearPreRankerConfig(w_base=1.0, w_cf=0.25, w_content=0.25, w_pop=0.1, w_cold=0.15, w_history=0.02),
        LinearPreRankerConfig(w_base=1.0, w_cf=0.35, w_content=0.2, w_pop=0.1, w_cold=0.1, w_history=0.01),
        LinearPreRankerConfig(w_base=1.0, w_cf=0.15, w_content=0.35, w_pop=0.08, w_cold=0.25, w_history=0.02),
        LinearPreRankerConfig(w_base=1.0, w_cf=0.1, w_content=0.4, w_pop=0.08, w_cold=0.35, w_history=0.02),
    ]

    best_cfg = grid[0]
    best_objective = -1.0
    best_overall_recall = 0.0
    best_cold_recall = 0.0
    for i, cfg in enumerate(grid, start=1):
        logger.event("STAGE2_TRY_START", trial=i, total_trials=len(grid), config=asdict(cfg))
        stage2_uc = PreRankCandidatesUseCase(preranker=LinearPreRanker(cfg=cfg))
        metrics = evaluate_prerank_metrics(
            users=val_users,
            gt_map=gt_map,
            seen_by_user=seen_by_user,
            cold_items_ids=cold,
            stage1_uc=stage1_uc,
            stage2_uc=stage2_uc,
            cmd=cmd,
        )
        objective = (
            0.45 * metrics["overall_recall"]
            + 0.55 * metrics["cold_recall"]
            + 0.10 * metrics["coverage_ratio"]
        )
        logger.event(
            "STAGE2_TRY_END",
            trial=i,
            total_trials=len(grid),
            overall_recall=round(metrics["overall_recall"], 6),
            cold_recall=round(metrics["cold_recall"], 6),
            cold_candidate_recall=round(metrics["cold_candidate_recall"], 6),
            coverage_ratio=round(metrics["coverage_ratio"], 6),
            objective=round(objective, 6),
        )
        if objective > best_objective:
            best_objective = objective
            best_cfg = cfg
            best_overall_recall = metrics["overall_recall"]
            best_cold_recall = metrics["cold_recall"]

    logger.progress("stage2_fit", done=1, total=1)
    logger.end_step(
        "stage2_fit",
        status="SUCCESS",
        best_objective=round(best_objective, 6),
        best_overall_recall=round(best_overall_recall, 6),
        best_cold_recall=round(best_cold_recall, 6),
        best_config=asdict(best_cfg),
    )
    return LinearPreRanker(cfg=best_cfg)


def evaluate_prerank_metrics(
    users: list[Any],
    gt_map: dict[Any, list[Any]],
    seen_by_user: dict[Any, set[Any]],
    cold_items_ids: set[Any],
    stage1_uc: GenerateCandidatesUseCase,
    stage2_uc: PreRankCandidatesUseCase,
    cmd: Any,
) -> dict[str, float]:
    hits = 0.0
    total = 0.0
    cold_hits = 0.0
    cold_total = 0.0
    cold_candidate_hits = 0.0
    cold_candidate_total = 0.0
    covered_items: set[Any] = set()
    for user_id in users:
        gt = gt_map.get(user_id, [])
        if not gt:
            continue
        seen = seen_by_user.get(user_id, set())
        candidates = stage1_uc.execute(
            GenerateCandidatesCommand(
                user_id=user_id,
                seen_items=seen,
                pool_size=cmd.candidate_pool_size,
                per_source_limit=cmd.candidate_per_source_limit,
                source_limits=_source_limits_for_stage1(
                    history_len=len(seen),
                    per_source_limit=cmd.candidate_per_source_limit,
                ),
            )
        )
        preranked = stage2_uc.execute(
            PreRankCandidatesCommand(
                user_id=user_id,
                candidates=candidates,
                history_len=len(seen),
                cold_item_ids=cold_items_ids,
                top_m=cmd.pre_top_m,
            )
        )
        pred = {x.item_id for x in preranked[: cmd.final_top_k]}
        covered_items.update(pred)
        gt_set = set(gt)
        hits += float(len(pred & gt_set))
        total += float(min(len(gt_set), cmd.final_top_k))

        gt_cold = gt_set & cold_items_ids
        if gt_cold:
            candidate_items = {x.item_id for x in candidates}
            cold_candidate_hits += float(len(candidate_items & gt_cold))
            cold_candidate_total += float(len(gt_cold))
            cold_hits += float(len(pred & gt_cold))
            cold_total += float(min(len(gt_cold), cmd.final_top_k))

    overall_recall = hits / total if total > 0 else 0.0
    cold_recall = cold_hits / cold_total if cold_total > 0 else 0.0
    cold_candidate_recall = cold_candidate_hits / cold_candidate_total if cold_candidate_total > 0 else 0.0
    coverage_ratio = float(len(covered_items) / max(1, cmd.candidate_pool_size))
    return {
        "overall_recall": overall_recall,
        "cold_recall": cold_recall,
        "cold_candidate_recall": cold_candidate_recall,
        "coverage_ratio": coverage_ratio,
    }


def _source_limits_for_stage1(history_len: int, per_source_limit: int) -> dict[str, int]:
    base = max(1, int(per_source_limit))
    if history_len <= 1:
        return {
            "cf": max(20, int(base * 0.25)),
            "content": int(base * 2.2),
            "pop": int(base * 1.2),
        }
    if history_len <= 5:
        return {
            "cf": max(40, int(base * 0.7)),
            "content": int(base * 1.8),
            "pop": int(base * 1.1),
        }
    return {
        "cf": base,
        "content": int(base * 1.25),
        "pop": base,
    }
