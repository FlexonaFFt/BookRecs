from __future__ import annotations

from dataclasses import asdict
from typing import Any

from source.application.use_cases.ranking.generate_candidates import (
    GenerateCandidatesCommand,
    GenerateCandidatesUseCase,
)
from source.application.use_cases.ranking.source_limits import source_limits_for_stage1
from source.application.use_cases.training.common.data_ops import build_seen_map, build_val_ground_truth, cold_items
from source.infrastructure.ranking.candidates import (
    ColdCandidateSource,
    CfCandidateSource,
    ContentCandidateSource,
    PopularCandidateSource,
)
from source.infrastructure.ranking.prerank import (
    CatBoostPreRanker,
    CatBoostPreRankerConfig,
    FeatureBuilder,
    LinearPreRanker,
)


def fit_stage2(data: dict[str, Any], stage1: dict[str, Any], cmd: Any, logger: Any) -> Any:
    logger.start_step("stage2_fit", total=3)
    val_users, gt_map = build_val_ground_truth(data["local_val"], limit=cmd.eval_users_limit)
    seen_by_user = build_seen_map(data["local_train"])
    cold = cold_items(data["local_train"], data["local_val"])

    candidate_sources = [
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
    ]
    stage1_uc = GenerateCandidatesUseCase(sources=candidate_sources, fallback_source=candidate_sources[-1])
    feature_builder = FeatureBuilder()
    logger.progress("stage2_fit", done=1, total=3)

    train_rows, eval_rows = build_prerank_dataset(
        users=val_users,
        gt_map=gt_map,
        seen_by_user=seen_by_user,
        cold_items_ids=cold,
        stage1_uc=stage1_uc,
        feature_builder=feature_builder,
        cmd=cmd,
    )
    logger.event(
        "STAGE2_DATASET",
        train_rows=len(train_rows),
        eval_rows=len(eval_rows),
        positives=sum(int(row["label"]) for row in train_rows),
        eval_positives=sum(int(row["label"]) for row in eval_rows),
    )
    logger.progress("stage2_fit", done=2, total=3)

    try:
        cfg = CatBoostPreRankerConfig(random_seed=cmd.seed)
        model = CatBoostPreRanker.fit(train_rows=train_rows, eval_rows=eval_rows, cfg=cfg)
        logger.progress("stage2_fit", done=3, total=3)
        logger.end_step(
            "stage2_fit",
            status="SUCCESS",
            model_type="catboost",
            train_rows=len(train_rows),
            eval_rows=len(eval_rows),
            config=asdict(cfg),
        )
        return model
    except Exception as exc:
        logger.event("STAGE2_FALLBACK_LINEAR", reason=str(exc))
        model = LinearPreRanker()
        logger.progress("stage2_fit", done=3, total=3)
        logger.end_step(
            "stage2_fit",
            status="SUCCESS",
            model_type="linear_fallback",
            train_rows=len(train_rows),
            eval_rows=len(eval_rows),
        )
        return model


def build_prerank_dataset(
    *,
    users: list[Any],
    gt_map: dict[Any, list[Any]],
    seen_by_user: dict[Any, set[Any]],
    cold_items_ids: set[Any],
    stage1_uc: GenerateCandidatesUseCase,
    feature_builder: FeatureBuilder,
    cmd: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    total_users = len(users)
    eval_start = int(total_users * 0.8)

    for idx, user_id in enumerate(users):
        gt = set(gt_map.get(user_id, []))
        if not gt:
            continue
        seen = seen_by_user.get(user_id, set())
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
            )
        )

        feature_rows = feature_builder.build(
            candidates=candidates,
            user_id=user_id,
            history_len=len(seen),
            cold_item_ids=cold_items_ids,
        )
        for row in feature_rows:
            payload = dict(row.features)
            payload["label"] = 1 if row.item_id in gt else 0
            payload["user_id"] = user_id
            rows.append(payload)

    if not rows:
        raise ValueError("No prerank training rows were generated")

    train_rows = rows[:]
    eval_rows: list[dict[str, Any]] = []
    if total_users >= 10:
        eval_user_ids = set(users[eval_start:])
        eval_rows = [row for row in rows if row["user_id"] in eval_user_ids]
        train_rows = [row for row in rows if row["user_id"] not in eval_user_ids]
        if not train_rows or not eval_rows:
            train_rows = rows[:]
            eval_rows = []

    for row in train_rows:
        row.pop("user_id", None)
    for row in eval_rows:
        row.pop("user_id", None)
    return train_rows, eval_rows
