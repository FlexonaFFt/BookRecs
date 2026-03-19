from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

from source.application.use_cases.training.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    TrainManifest,
    build_layout,
)
from source.application.use_cases.training.common.data_ops import load_dataset
from source.application.use_cases.training.common.publish import publish_local_artifacts
from source.application.use_cases.training.pipeline.models import (
    TrainPipelineCommand,
    TrainPipelineResult,
)
from source.application.use_cases.training.stages.evaluate import evaluate_pipeline
from source.application.use_cases.training.stages.stage1_fit import fit_stage1
from source.application.use_cases.training.stages.stage2_fit import fit_stage2
from source.application.use_cases.training.stages.stage3_fit import fit_stage3
from source.infrastructure.training import TrainLogger


# Реализует сценарий пайплайна обучения.
class TrainPipelineUseCase:
    def execute(self, cmd: TrainPipelineCommand) -> TrainPipelineResult:
        if pd is None:
            raise RuntimeError(
                "pandas is required for training. Install project dependencies first."
            )
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

        data = load_dataset(pd, cmd.dataset_dir)
        logger.event(
            "DATA_LOADED",
            train_rows=int(len(data["local_train"])),
            val_rows=int(len(data["local_val"])),
            books_rows=int(len(data["books"])),
        )

        timings: dict[str, float] = {}

        step_started = time.time()
        stage1 = fit_stage1(data=data, cmd=cmd, logger=logger)
        timings["stage1_fit"] = round(time.time() - step_started, 3)
        logger.event(
            "STEP_TIMING", step="stage1_fit", duration_sec=timings["stage1_fit"]
        )

        step_started = time.time()
        stage2_model = fit_stage2(data=data, stage1=stage1, cmd=cmd, logger=logger)
        timings["stage2_fit"] = round(time.time() - step_started, 3)
        logger.event(
            "STEP_TIMING", step="stage2_fit", duration_sec=timings["stage2_fit"]
        )

        step_started = time.time()
        stage3_model = fit_stage3(
            data=data,
            stage1=stage1,
            stage2_model=stage2_model,
            cmd=cmd,
            logger=logger,
        )
        timings["stage3_fit"] = round(time.time() - step_started, 3)
        logger.event(
            "STEP_TIMING", step="stage3_fit", duration_sec=timings["stage3_fit"]
        )

        step_started = time.time()
        metrics = evaluate_pipeline(
            data=data,
            stage1=stage1,
            stage2_model=stage2_model,
            stage3_model=stage3_model,
            cmd=cmd,
            logger=logger,
        )
        timings["evaluate"] = round(time.time() - step_started, 3)
        logger.event("STEP_TIMING", step="evaluate", duration_sec=timings["evaluate"])

        step_started = time.time()
        publish_local_artifacts(
            layout=layout,
            stage1=stage1,
            stage2_model=stage2_model,
            stage3_model=stage3_model,
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

        layout.metrics.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        layout.timings.write_text(
            json.dumps(timings, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        layout.manifest.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.event(
            "RUN_END", status="SUCCESS", duration_sec=duration_sec, metrics=metrics
        )
        return TrainPipelineResult(
            run_id=run_id,
            run_dir=str(run_dir),
            manifest_path=str(layout.manifest),
            metrics_path=str(layout.metrics),
            timings_path=str(layout.timings),
            log_path=str(layout.train_log),
        )
