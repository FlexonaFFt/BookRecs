from __future__ import annotations

import itertools
import json
import math
import pickle
import random
import time

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

from source.application.use_cases.ranking.final_rank import FinalRankUseCase
from source.application.use_cases.ranking.generate_candidates import GenerateCandidatesUseCase
from source.application.use_cases.ranking.prerank_candidates import PreRankCandidatesUseCase
from source.application.use_cases.ranking.reco_flow import RecoFlowCommand, RecoFlowUseCase
from source.application.use_cases.training.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactLayout,
    TrainManifest,
    build_layout,
)
from source.infrastructure.processing.postprocessing import PostprocessTemplate
from source.infrastructure.ranking.candidates import SourceCf, SourceContent, SourcePop
from source.infrastructure.ranking.finalrank import RankerTemplate
from source.infrastructure.ranking.prerank import PreRankLinear, PreRankLinearConfig
from source.infrastructure.training import TrainLogger


@dataclass(frozen=True)
class TrainPipelineCommand:
    dataset_dir: str = "artifacts/tmp_preprocessed/goodreads_ya"
    output_root: str = "artifacts/runs"
    run_name: str | None = None
    eval_users_limit: int = 2000
    candidate_pool_size: int = 1000
    candidate_per_source_limit: int = 300
    pre_top_m: int = 300
    final_top_k: int = 10
    cf_max_neighbors: int = 120
    content_max_neighbors: int = 120
    seed: int = 42


@dataclass(frozen=True)
class TrainPipelineResult:
    run_id: str
    run_dir: str
    manifest_path: str
    metrics_path: str
    timings_path: str
    log_path: str


class TrainPipelineUseCase:
    def execute(self, cmd: TrainPipelineCommand) -> TrainPipelineResult:
        if pd is None:
            raise RuntimeError("pandas is required for training. Install project dependencies first.")
        if cmd.eval_users_limit <= 0:
            raise ValueError("eval_users_limit must be > 0")
        if cmd.final_top_k <= 0:
            raise ValueError("final_top_k must be > 0")

        random.seed(cmd.seed)

        run_id = cmd.run_name or str(uuid4())
        run_dir = Path(cmd.output_root) / run_id
        layout = build_layout(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        layout.model_dir.mkdir(parents=True, exist_ok=True)

        logger = TrainLogger(run_id=run_id, log_file=layout.train_log)
        pipeline_started = time.time()
        logger.event(
            "RUN_START",
            run_dir=str(run_dir),
            dataset_dir=cmd.dataset_dir,
            config=asdict(cmd),
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        )

        data = self._load_dataset(cmd.dataset_dir)
        logger.event(
            "DATA_LOADED",
            train_rows=int(len(data["local_train"])),
            val_rows=int(len(data["local_val"])),
            books_rows=int(len(data["books"])),
        )

        timings: dict[str, float] = {}

        # Stage 1 fit
        step_started = time.time()
        stage1 = self._fit_stage1(data=data, cmd=cmd, logger=logger)
        timings["stage1_fit"] = round(time.time() - step_started, 3)
        logger.event("STEP_TIMING", step="stage1_fit", duration_sec=timings["stage1_fit"])

        # Stage 2 fit (weight search)
        step_started = time.time()
        stage2_cfg = self._fit_stage2(data=data, stage1=stage1, cmd=cmd, logger=logger)
        timings["stage2_fit"] = round(time.time() - step_started, 3)
        logger.event("STEP_TIMING", step="stage2_fit", duration_sec=timings["stage2_fit"])

        # Stage 3 fit (template; no extra params for now)
        step_started = time.time()
        stage3_cfg = self._fit_stage3(logger=logger)
        timings["stage3_fit"] = round(time.time() - step_started, 3)
        logger.event("STEP_TIMING", step="stage3_fit", duration_sec=timings["stage3_fit"])

        # Evaluate
        step_started = time.time()
        metrics = self._evaluate(data=data, stage1=stage1, stage2_cfg=stage2_cfg, cmd=cmd, logger=logger)
        timings["evaluate"] = round(time.time() - step_started, 3)
        logger.event("STEP_TIMING", step="evaluate", duration_sec=timings["evaluate"])

        # Publish local artifacts
        step_started = time.time()
        self._publish_local(
            layout=layout,
            stage1=stage1,
            stage2_cfg=stage2_cfg,
            stage3_cfg=stage3_cfg,
            metrics=metrics,
            logger=logger,
        )
        timings["publish"] = round(time.time() - step_started, 3)
        logger.event("STEP_TIMING", step="publish", duration_sec=timings["publish"])

        duration_sec = round(time.time() - pipeline_started, 3)
        manifest = TrainManifest.build(
            run_id=run_id,
            status="SUCCESS",
            duration_sec=duration_sec,
            dataset_dir=cmd.dataset_dir,
            config=asdict(cmd),
            layout=layout,
            metrics=metrics,
            timings=timings,
        )

        layout.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        layout.timings.write_text(json.dumps(timings, ensure_ascii=False, indent=2), encoding="utf-8")
        layout.manifest.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

        logger.event("RUN_END", status="SUCCESS", duration_sec=duration_sec, metrics=metrics)
        return TrainPipelineResult(
            run_id=run_id,
            run_dir=str(run_dir),
            manifest_path=str(layout.manifest),
            metrics_path=str(layout.metrics),
            timings_path=str(layout.timings),
            log_path=str(layout.train_log),
        )

    @staticmethod
    def _load_dataset(dataset_dir: str) -> dict[str, Any]:
        root = Path(dataset_dir)
        required = {
            "books": root / "books.parquet",
            "local_train": root / "local_train.parquet",
            "local_val": root / "local_val.parquet",
        }
        for name, path in required.items():
            if not path.exists():
                raise FileNotFoundError(f"{name} file not found: {path}")
        return {
            "books": pd.read_parquet(required["books"]),
            "local_train": pd.read_parquet(required["local_train"]),
            "local_val": pd.read_parquet(required["local_val"]),
        }

    def _fit_stage1(self, data: dict[str, Any], cmd: TrainPipelineCommand, logger: TrainLogger) -> dict[str, Any]:
        train = data["local_train"]
        books = data["books"]
        logger.start_step("stage1_fit", total=3)

        pop_items, pop_scores = self._fit_popularity(train)
        logger.progress("stage1_fit", done=1, total=3)

        cf_neighbors = self._fit_cf_neighbors(
            interactions=train,
            max_neighbors=cmd.cf_max_neighbors,
            logger=logger,
        )
        logger.progress("stage1_fit", done=2, total=3)

        content_similar = self._fit_content_neighbors(
            books=books,
            max_neighbors=cmd.content_max_neighbors,
            logger=logger,
        )
        logger.progress("stage1_fit", done=3, total=3)
        logger.end_step(
            "stage1_fit",
            status="SUCCESS",
            pop_items=len(pop_items),
            cf_items=len(cf_neighbors),
            content_items=len(content_similar),
        )
        return {
            "pop_items": pop_items,
            "pop_scores": pop_scores,
            "cf_neighbors": cf_neighbors,
            "content_similar": content_similar,
        }

    def _fit_stage2(
        self,
        data: dict[str, Any],
        stage1: dict[str, Any],
        cmd: TrainPipelineCommand,
        logger: TrainLogger,
    ) -> PreRankLinearConfig:
        logger.start_step("stage2_fit", total=1)
        val_users, gt_map = self._build_val_ground_truth(data["local_val"], limit=cmd.eval_users_limit)
        seen_by_user = self._build_seen_map(data["local_train"])
        cold_items = self._cold_items(data["local_train"], data["local_val"])

        candidate_sources = [
            SourceCf(stage1["cf_neighbors"]),
            SourceContent(stage1["content_similar"]),
            SourcePop(stage1["pop_items"], stage1["pop_scores"]),
        ]
        stage1_uc = GenerateCandidatesUseCase(sources=candidate_sources, fallback_source=candidate_sources[-1])

        # Tiny grid search over pre-rank weights.
        grid = [
            PreRankLinearConfig(w_base=1.0, w_cf=0.2, w_content=0.2, w_pop=0.1, w_cold=0.1, w_history=0.02),
            PreRankLinearConfig(w_base=1.0, w_cf=0.3, w_content=0.2, w_pop=0.1, w_cold=0.1, w_history=0.02),
            PreRankLinearConfig(w_base=1.0, w_cf=0.25, w_content=0.25, w_pop=0.1, w_cold=0.15, w_history=0.02),
            PreRankLinearConfig(w_base=1.0, w_cf=0.35, w_content=0.2, w_pop=0.1, w_cold=0.1, w_history=0.01),
        ]

        best_cfg = grid[0]
        best_recall = -1.0
        for i, cfg in enumerate(grid, start=1):
            logger.event("STAGE2_TRY_START", trial=i, total_trials=len(grid), config=asdict(cfg))
            stage2_uc = PreRankCandidatesUseCase(preranker=PreRankLinear(cfg=cfg))
            recall = self._evaluate_prerank_recall(
                users=val_users,
                gt_map=gt_map,
                seen_by_user=seen_by_user,
                cold_items=cold_items,
                stage1_uc=stage1_uc,
                stage2_uc=stage2_uc,
                cmd=cmd,
            )
            logger.event("STAGE2_TRY_END", trial=i, total_trials=len(grid), recall=round(recall, 6))
            if recall > best_recall:
                best_recall = recall
                best_cfg = cfg

        logger.progress("stage2_fit", done=1, total=1)
        logger.end_step(
            "stage2_fit",
            status="SUCCESS",
            best_recall=round(best_recall, 6),
            best_config=asdict(best_cfg),
        )
        return best_cfg

    @staticmethod
    def _fit_stage3(logger: TrainLogger) -> dict[str, Any]:
        # Stage-3 template has no trainable params yet.
        logger.start_step("stage3_fit", total=1)
        logger.progress("stage3_fit", done=1, total=1)
        cfg = {"ranker": "RankerTemplate", "postprocessor": "PostprocessTemplate"}
        logger.end_step("stage3_fit", status="SUCCESS", config=cfg)
        return cfg

    def _evaluate(
        self,
        data: dict[str, Any],
        stage1: dict[str, Any],
        stage2_cfg: PreRankLinearConfig,
        cmd: TrainPipelineCommand,
        logger: TrainLogger,
    ) -> dict[str, float]:
        logger.start_step("evaluate", total=1)
        val_users, gt_map = self._build_val_ground_truth(data["local_val"], limit=cmd.eval_users_limit)
        seen_by_user = self._build_seen_map(data["local_train"])
        cold_items = self._cold_items(data["local_train"], data["local_val"])

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
                    cold_item_ids=cold_items,
                    candidate_pool_size=cmd.candidate_pool_size,
                    candidate_per_source_limit=cmd.candidate_per_source_limit,
                    pre_top_m=cmd.pre_top_m,
                    final_top_k=cmd.final_top_k,
                )
            )
            pred = [x.item_id for x in result.final_items]
            covered_items.update(pred[: cmd.final_top_k])
            ndcg_scores.append(self._ndcg_at_k(pred, gt_items, cmd.final_top_k))
            recall_scores.append(self._recall_at_k(pred, gt_items, cmd.final_top_k))

            gt_cold = [x for x in gt_items if x in cold_items]
            if gt_cold:
                cold_ndcg_scores.append(self._ndcg_at_k(pred, gt_cold, cmd.final_top_k))
                cold_recall_scores.append(self._recall_at_k(pred, gt_cold, cmd.final_top_k))

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

    @staticmethod
    def _publish_local(
        layout: ArtifactLayout,
        stage1: dict[str, Any],
        stage2_cfg: PreRankLinearConfig,
        stage3_cfg: dict[str, Any],
        metrics: dict[str, float],
        logger: TrainLogger,
    ) -> None:
        logger.start_step("publish", total=4)

        with layout.stage1_model.open("wb") as f:
            pickle.dump(stage1, f)
        logger.progress("publish", done=1, total=4)

        layout.stage2_config.write_text(
            json.dumps(asdict(stage2_cfg), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.progress("publish", done=2, total=4)

        layout.stage3_config.write_text(
            json.dumps(stage3_cfg, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.progress("publish", done=3, total=4)

        layout.metrics_snapshot.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.progress("publish", done=4, total=4)
        logger.end_step("publish", status="SUCCESS", model_dir=str(layout.model_dir))

    @staticmethod
    def _fit_popularity(train: Any) -> tuple[list[Any], dict[Any, float]]:
        pop = train.groupby("item_id", as_index=False).size().rename(columns={"size": "n"})
        pop = pop.sort_values("n", ascending=False).reset_index(drop=True)
        top_items = pop["item_id"].tolist()
        max_n = float(pop["n"].max()) if len(pop) else 1.0
        if max_n <= 0:
            max_n = 1.0
        scores = {row.item_id: float(row.n / max_n) for row in pop.itertuples(index=False)}
        return top_items, scores

    @staticmethod
    def _fit_cf_neighbors(interactions: Any, max_neighbors: int, logger: TrainLogger) -> dict[Any, list[tuple[Any, float]]]:
        # Co-occurrence + cosine-ish score.
        user_items = (
            interactions[["user_id", "item_id"]]
            .drop_duplicates(["user_id", "item_id"])
            .groupby("user_id", sort=False)["item_id"]
            .agg(list)
        )
        pair_counts: Counter[tuple[Any, Any]] = Counter()
        item_counts: Counter[Any] = Counter()

        users_total = len(user_items)
        for i, items in enumerate(user_items.tolist(), start=1):
            uniq = list(dict.fromkeys(items))
            item_counts.update(uniq)
            for a, b in itertools.combinations(sorted(uniq), 2):
                pair_counts[(a, b)] += 1
            if i % max(1, users_total // 20) == 0 or i == users_total:
                logger.event("STAGE1_CF_PROGRESS", done=i, total=users_total)

        neighbors: dict[Any, list[tuple[Any, float]]] = defaultdict(list)
        for (a, b), co in pair_counts.items():
            denom = math.sqrt(float(item_counts[a]) * float(item_counts[b]))
            if denom <= 0:
                continue
            score = float(co / denom)
            neighbors[a].append((b, score))
            neighbors[b].append((a, score))

        out: dict[Any, list[tuple[Any, float]]] = {}
        for item_id, vals in neighbors.items():
            vals.sort(key=lambda x: x[1], reverse=True)
            out[item_id] = vals[:max_neighbors]
        return out

    @staticmethod
    def _fit_content_neighbors(books: Any, max_neighbors: int, logger: TrainLogger) -> dict[Any, list[tuple[Any, float]]]:
        # Build sparse similarity via shared authors/series/tags.
        data = books.copy()
        for col in ["authors", "series", "tags"]:
            if col not in data.columns:
                data[col] = [[] for _ in range(len(data))]
            data[col] = data[col].apply(lambda x: x if isinstance(x, list) else [])

        by_item = data[["item_id", "authors", "series", "tags"]].drop_duplicates("item_id").reset_index(drop=True)

        author_index: dict[str, list[Any]] = defaultdict(list)
        series_index: dict[str, list[Any]] = defaultdict(list)
        tag_index: dict[str, list[Any]] = defaultdict(list)

        total = len(by_item)
        for i, row in enumerate(by_item.itertuples(index=False), start=1):
            item_id = row.item_id
            for a in list(dict.fromkeys(row.authors))[:8]:
                author_index[str(a)].append(item_id)
            for s in list(dict.fromkeys(row.series))[:4]:
                series_index[str(s)].append(item_id)
            for t in list(dict.fromkeys(row.tags))[:20]:
                tag_index[str(t)].append(item_id)
            if i % max(1, total // 20) == 0 or i == total:
                logger.event("STAGE1_CONTENT_INDEX_PROGRESS", done=i, total=total)

        out: dict[Any, list[tuple[Any, float]]] = {}
        for i, row in enumerate(by_item.itertuples(index=False), start=1):
            item_id = row.item_id
            score_map: dict[Any, float] = defaultdict(float)

            for a in list(dict.fromkeys(row.authors))[:8]:
                for other in author_index.get(str(a), []):
                    if other != item_id:
                        score_map[other] += 2.0
            for s in list(dict.fromkeys(row.series))[:4]:
                for other in series_index.get(str(s), []):
                    if other != item_id:
                        score_map[other] += 2.0
            for t in list(dict.fromkeys(row.tags))[:20]:
                for other in tag_index.get(str(t), []):
                    if other != item_id:
                        score_map[other] += 0.5

            ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:max_neighbors]
            out[item_id] = [(oid, float(score)) for oid, score in ranked]
            if i % max(1, total // 20) == 0 or i == total:
                logger.event("STAGE1_CONTENT_SCORE_PROGRESS", done=i, total=total)
        return out

    @staticmethod
    def _build_seen_map(train: Any) -> dict[Any, set[Any]]:
        return (
            train.groupby("user_id", sort=False)["item_id"]
            .agg(lambda x: set(x.tolist()))
            .to_dict()
        )

    @staticmethod
    def _cold_items(train: Any, val: Any) -> set[Any]:
        train_items = set(train["item_id"].dropna().tolist())
        val_items = set(val["item_id"].dropna().tolist())
        return val_items - train_items

    @staticmethod
    def _build_val_ground_truth(val: Any, limit: int) -> tuple[list[Any], dict[Any, list[Any]]]:
        grouped = (
            val.groupby("user_id", sort=False)["item_id"]
            .agg(list)
            .reset_index()
        )
        users = grouped["user_id"].tolist()[:limit]
        gt_map = {row.user_id: list(row.item_id) for row in grouped.itertuples(index=False) if row.user_id in set(users)}
        return users, gt_map

    def _evaluate_prerank_recall(
        self,
        users: list[Any],
        gt_map: dict[Any, list[Any]],
        seen_by_user: dict[Any, set[Any]],
        cold_items: set[Any],
        stage1_uc: GenerateCandidatesUseCase,
        stage2_uc: PreRankCandidatesUseCase,
        cmd: TrainPipelineCommand,
    ) -> float:
        hits = 0.0
        total = 0.0
        for user_id in users:
            gt = gt_map.get(user_id, [])
            if not gt:
                continue
            seen = seen_by_user.get(user_id, set())
            cands = stage1_uc.execute(
                cmd=type("X", (), {
                    "user_id": user_id,
                    "seen_items": seen,
                    "pool_size": cmd.candidate_pool_size,
                    "per_source_limit": cmd.candidate_per_source_limit,
                })()
            )
            preranked = stage2_uc.execute(
                cmd=type("Y", (), {
                    "user_id": user_id,
                    "candidates": cands,
                    "history_len": len(seen),
                    "cold_item_ids": cold_items,
                    "top_m": cmd.pre_top_m,
                })()
            )
            pred = {x.item_id for x in preranked[: cmd.final_top_k]}
            gt_set = set(gt)
            hits += float(len(pred & gt_set))
            total += float(min(len(gt_set), cmd.final_top_k))
        if total <= 0:
            return 0.0
        return hits / total

    @staticmethod
    def _recall_at_k(pred_items: list[Any], gt_items: list[Any], k: int) -> float:
        if not gt_items:
            return 0.0
        gt_set = set(gt_items)
        hits = sum(1 for x in pred_items[:k] if x in gt_set)
        denom = min(len(gt_set), k)
        return float(hits / denom) if denom > 0 else 0.0

    @staticmethod
    def _ndcg_at_k(pred_items: list[Any], gt_items: list[Any], k: int) -> float:
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
