from __future__ import annotations

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
from source.infrastructure.ranking.finalrank import LinearFinalReranker, LinearFinalRerankerConfig


# Обучает линейный финальный реранкер на валидационном срезе.
def fit_stage3(
    data: dict[str, Any],
    stage1: dict[str, Any],
    stage2_model: Any,
    cmd: Any,
    logger: Any,
) -> LinearFinalReranker:
    logger.start_step("stage3_fit", total=1)
    val_users, gt_map = build_val_ground_truth(data["local_val"], limit=cmd.eval_users_limit)
    seen_by_user = build_seen_map(data["local_train"])
    cold = cold_items(data["local_train"], data["local_val"])

    stage1_uc = GenerateCandidatesUseCase(
        sources=[
            CfCandidateSource(stage1["cf_neighbors"]),
            ContentCandidateSource(stage1["content_similar"]),
            PopularCandidateSource(stage1["pop_items"], stage1["pop_scores"]),
        ],
        fallback_source=PopularCandidateSource(stage1["pop_items"], stage1["pop_scores"]),
    )
    stage2_uc = PreRankCandidatesUseCase(preranker=stage2_model)

    stats: dict[str, dict[str, float]] = {}
    total_rows = 0.0
    total_hits = 0.0
    for user_id in val_users:
        seen = seen_by_user.get(user_id, set())
        gt = set(gt_map.get(user_id, []))
        if not gt:
            continue

        candidates = stage1_uc.execute(
            GenerateCandidatesCommand(
                user_id=user_id,
                seen_items=seen,
                pool_size=cmd.candidate_pool_size,
                per_source_limit=cmd.candidate_per_source_limit,
            )
        )
        preranked = stage2_uc.execute(
            PreRankCandidatesCommand(
                user_id=user_id,
                candidates=candidates,
                history_len=len(seen),
                cold_item_ids=cold,
                top_m=cmd.pre_top_m,
            )
        )

        for item in preranked[: cmd.pre_top_m]:
            source_stats = stats.setdefault(item.source, {"rows": 0.0, "hits": 0.0})
            source_stats["rows"] += 1.0
            total_rows += 1.0
            if item.item_id in gt:
                source_stats["hits"] += 1.0
                total_hits += 1.0

    global_rate = total_hits / total_rows if total_rows > 0 else 0.0
    source_bias: dict[str, float] = {}
    for source, source_stats in stats.items():
        rows = source_stats["rows"]
        rate = source_stats["hits"] / rows if rows > 0 else 0.0
        source_bias[source] = round((rate - global_rate) * 0.35, 6)

    cfg = LinearFinalRerankerConfig(source_bias=source_bias)
    model = LinearFinalReranker(cfg=cfg)
    logger.progress("stage3_fit", done=1, total=1)
    logger.end_step(
        "stage3_fit",
        status="SUCCESS",
        global_rate=round(global_rate, 6),
        source_bias=source_bias,
    )
    return model
