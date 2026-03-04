from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from typing import Any

from source.application.use_cases.training.artifacts import ArtifactLayout


def publish_local_artifacts(
    layout: ArtifactLayout,
    stage1: dict[str, Any],
    stage2_cfg: Any,
    stage3_cfg: dict[str, Any],
    metrics: dict[str, float],
    logger: Any,
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
