from __future__ import annotations

from typing import Any

from source.application.use_cases.ranking.final_rank import FinalRankUseCase
from source.application.use_cases.ranking.generate_candidates import GenerateCandidatesUseCase
from source.application.use_cases.ranking.prerank_candidates import PreRankCandidatesUseCase
from source.application.use_cases.ranking.reco_flow import RecoFlowCommand, RecoFlowUseCase
from source.application.use_cases.training.common.data_ops import build_seen_map, build_val_ground_truth, cold_items
from source.application.use_cases.training.common.metrics import ndcg_at_k, recall_at_k
from source.infrastructure.processing.postprocessing import PostprocessTemplate
from source.infrastructure.ranking.candidates import SourceCf, SourceContent, SourcePop
from source.infrastructure.ranking.finalrank import RankerTemplate
from source.infrastructure.ranking.prerank import PreRankLinear


def evaluate_pipeline(
    data: dict[str, Any],
    stage1: dict[str, Any],
    stage2_cfg: Any,
    cmd: Any,
    logger: Any,
) -> dict[str, float]:
    logger.start_step("evaluate", total=1)
    val_users, gt_map = build_val_ground_truth(data["local_val"], limit=cmd.eval_users_limit)
    seen_by_user = build_seen_map(data["local_train"])
    cold = cold_items(data["local_train"], data["local_val"])

    stage1_uc = GenerateCandidatesUseCase(
        sources=[
            SourceCf(stage1["cf_neighbors"]),
            SourceContent(stage1["content_similar"]),
            SourcePop(stage1["pop_items"], stage1["pop_scores"]),
        ],
        fallback_source=SourcePop(stage1["pop_items"], stage1["pop_scores"]),
    )
    stage2_uc = PreRankCandidatesUseCase(preranker=PreRankLinear(cfg=stage2_cfg))
    stage3_uc = FinalRankUseCase(RankerTemplate(), PostprocessTemplate())
    flow = RecoFlowUseCase(stage1=stage1_uc, stage2=stage2_uc, stage3=stage3_uc)

    ndcg_scores: list[float] = []
    recall_scores: list[float] = []
    cold_ndcg_scores: list[float] = []
    cold_recall_scores: list[float] = []
    covered_items: set[Any] = set()

    total = len(val_users)
    for i, user_id in enumerate(val_users, start=1):
        seen = seen_by_user.get(user_id, set())
        gt_items = gt_map.get(user_id, [])
        if not gt_items:
            continue

        result = flow.execute(
            RecoFlowCommand(
                user_id=user_id,
                seen_items=seen,
                history_len=len(seen),
                cold_item_ids=cold,
                candidate_pool_size=cmd.candidate_pool_size,
                candidate_per_source_limit=cmd.candidate_per_source_limit,
                pre_top_m=cmd.pre_top_m,
                final_top_k=cmd.final_top_k,
            )
        )
        pred = [x.item_id for x in result.final_items]
        covered_items.update(pred[: cmd.final_top_k])
        ndcg_scores.append(ndcg_at_k(pred, gt_items, cmd.final_top_k))
        recall_scores.append(recall_at_k(pred, gt_items, cmd.final_top_k))

        gt_cold = [x for x in gt_items if x in cold]
        if gt_cold:
            cold_ndcg_scores.append(ndcg_at_k(pred, gt_cold, cmd.final_top_k))
            cold_recall_scores.append(recall_at_k(pred, gt_cold, cmd.final_top_k))

        if i % max(1, total // 20) == 0 or i == total:
            logger.progress("evaluate", done=i, total=total)

    catalog_size = max(1, int(data["books"]["item_id"].nunique()))
    metrics = {
        f"ndcg@{cmd.final_top_k}": float(sum(ndcg_scores) / max(1, len(ndcg_scores))),
        f"recall@{cmd.final_top_k}": float(sum(recall_scores) / max(1, len(recall_scores))),
        f"coverage@{cmd.final_top_k}": float(len(covered_items) / catalog_size),
        f"cold_ndcg@{cmd.final_top_k}": float(sum(cold_ndcg_scores) / max(1, len(cold_ndcg_scores))),
        f"cold_recall@{cmd.final_top_k}": float(sum(cold_recall_scores) / max(1, len(cold_recall_scores))),
    }
    logger.end_step("evaluate", status="SUCCESS", metrics=metrics)
    return metrics
