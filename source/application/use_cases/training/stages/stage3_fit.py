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
from source.application.use_cases.ranking.source_limits import (
    source_limits_for_stage1,
    source_min_quota_for_stage1,
)
from source.application.use_cases.training.common.data_ops import build_seen_map, build_val_ground_truth, cold_items
from source.infrastructure.ranking.candidates import (
    ColdCandidateSource,
    CfCandidateSource,
    ContentCandidateSource,
    PopularCandidateSource,
)
from source.infrastructure.ranking.finalrank import PolicyFinalReranker, PolicyFinalRerankerConfig


# Обучает policy-based финальный реранкер на валидационном срезе.
def fit_stage3(
    data: dict[str, Any],
    stage1: dict[str, Any],
    stage2_model: Any,
    cmd: Any,
    logger: Any,
) -> PolicyFinalReranker:
    logger.start_step("stage3_fit", total=1)
    val_users, gt_map = build_val_ground_truth(data["local_val"], limit=cmd.eval_users_limit)
    seen_by_user = build_seen_map(data["local_train"])
    cold = cold_items(data["local_train"], data["local_val"])

    stage1_uc = GenerateCandidatesUseCase(
        sources=[
            CfCandidateSource(stage1["cf_neighbors"]),
            ContentCandidateSource(
                stage1["content_similar"],
                popularity_scores=stage1["pop_scores"],
            ),
            ColdCandidateSource(
                item_metadata=stage1["item_metadata"],
                author_index=stage1["author_index"],
                series_index=stage1["series_index"],
                tag_index=stage1["tag_index"],
                popularity_scores=stage1["pop_scores"],
            ),
            PopularCandidateSource(stage1["pop_items"], stage1["pop_scores"]),
        ],
        fallback_source=PopularCandidateSource(stage1["pop_items"], stage1["pop_scores"]),
    )
    stage2_uc = PreRankCandidatesUseCase(preranker=stage2_model)

    stats: dict[str, dict[str, float]] = {}
    total_rows = 0.0
    total_hits = 0.0
    total_cold_rows = 0.0
    total_cold_hits = 0.0
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
                source_limits=source_limits_for_stage1(
                    history_len=len(seen),
                    per_source_limit=cmd.candidate_per_source_limit,
                ),
                source_min_quota=source_min_quota_for_stage1(
                    history_len=len(seen),
                    pool_size=cmd.candidate_pool_size,
                ),
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
            source_stats = stats.setdefault(
                item.source,
                {"rows": 0.0, "hits": 0.0, "cold_rows": 0.0, "cold_hits": 0.0},
            )
            source_stats["rows"] += 1.0
            total_rows += 1.0
            is_cold = bool(item.item_id in cold)
            if is_cold:
                source_stats["cold_rows"] += 1.0
                total_cold_rows += 1.0
            if item.item_id in gt:
                source_stats["hits"] += 1.0
                total_hits += 1.0
                if is_cold:
                    source_stats["cold_hits"] += 1.0
                    total_cold_hits += 1.0

    global_rate = total_hits / total_rows if total_rows > 0 else 0.0
    global_cold_rate = total_cold_hits / total_cold_rows if total_cold_rows > 0 else 0.0
    source_bias: dict[str, float] = {}
    for source, source_stats in stats.items():
        rows = source_stats["rows"]
        cold_rows = source_stats["cold_rows"]
        rate = source_stats["hits"] / rows if rows > 0 else 0.0
        cold_rate = source_stats["cold_hits"] / cold_rows if cold_rows > 0 else 0.0
        score = 0.25 * (rate - global_rate) + 0.65 * (cold_rate - global_cold_rate)
        source_bias[source] = round(score, 6)

    target_cold_items = 1 if total_cold_rows > 0 and cmd.final_top_k >= 5 else 0
    cfg = PolicyFinalRerankerConfig(
        source_bias=source_bias,
        source_repeat_penalty=0.04,
        cold_item_boost=0.08 if global_cold_rate > 0 else 0.0,
        metadata_overlap_boost=0.05,
        popularity_penalty=0.025,
        target_cold_items=target_cold_items,
    )
    model = PolicyFinalReranker(cfg=cfg)
    logger.progress("stage3_fit", done=1, total=1)
    logger.end_step(
        "stage3_fit",
        status="SUCCESS",
        global_rate=round(global_rate, 6),
        global_cold_rate=round(global_cold_rate, 6),
        source_bias=source_bias,
        target_cold_items=target_cold_items,
    )
    return model
